import collections
import operator

import torch
from torch._dynamo.utils import counters

from ..pattern_matcher import CallFunctionVarArgs

aten = torch.ops.aten


def _compute_graph_nodes_dependency_matrix(graph):
    nodes = list(graph.nodes)
    num_nodes = len(nodes)

    dependency_matrix = [[False] * num_nodes for _ in range(num_nodes)]

    def i_depends_on_j(i, j):
        return nodes[j] in nodes[i].all_input_nodes

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dependency_matrix[i][j] = i_depends_on_j(i, j)
            else:
                dependency_matrix[i][j] = True

    # Floyd Warshall Algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                dependency_matrix[i][j] = dependency_matrix[i][j] or (
                    dependency_matrix[i][k] and dependency_matrix[k][j]
                )

    return {
        nodes[i]: {nodes[j]: dependent for j, dependent in enumerate(row)}
        for i, row in enumerate(dependency_matrix)
    }


def _get_independent_node_subsets(mm_node_list, dependency_matrix):
    """
    Return an iterator of node subsets, each subset only contains nodes
    those are independent with each other.
    """

    def _is_independent(n1, n2):
        if n1 != n2 and (dependency_matrix[n1][n2] or dependency_matrix[n2][n1]):
            return False
        return True

    while len(mm_node_list) > 0:
        next_round_mm_node_list = []

        independent_set = []

        for mm_node in mm_node_list:
            if all(_is_independent(mm_node, e) for e in independent_set):
                independent_set.append(mm_node)
            else:
                next_round_mm_node_list.append(mm_node)

        yield independent_set

        mm_node_list = next_round_mm_node_list


class GroupBatchFusionBase:
    def match(self, node):
        raise NotImplementedError("match called on base")

    def fuse(self, graph, subset):
        raise NotImplementedError("fuse called on base")


class GroupFusion(GroupBatchFusionBase):
    pass


class BatchFusion(GroupBatchFusionBase):
    pass


class GroupLinearFusion(GroupFusion):
    def match(self, node):
        if (
            CallFunctionVarArgs(aten.mm.default).match(node)
            and len(node.args[0].meta["tensor_meta"].shape) == 2
        ):
            group_key = ("group_linear",)
        elif (
            CallFunctionVarArgs(aten.addmm.default).match(node)
            and len(node.args[1].meta["tensor_meta"].shape) == 2
            and node.kwargs.get("beta", 1.0) == 1.0
            and node.kwargs.get("alpha", 1.0) == 1.0
        ):
            group_key = ("group_linear",)
        else:
            group_key = None
        return group_key

    def fuse(self, graph, subset):
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            else:
                assert CallFunctionVarArgs(aten.mm.default).match(node)
                input, weight = node.args
                bias = None
            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        if all(bias is None for bias in group_biases):
            group_biases = None

        with graph.inserting_before(subset[0]):
            fused_mm = graph.call_function(
                torch.ops.fbgemm.group_gemm,
                args=(group_inputs, group_weights, group_biases),
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)


def apply_group_batch_fusion(graph, fusion_rule):
    fusible_groups = collections.defaultdict(list)

    dependency_matrix = _compute_graph_nodes_dependency_matrix(graph)

    for node in graph.nodes:
        group_key = fusion_rule.match(node)
        if group_key:
            fusible_groups[group_key].append(node)

    for fusible_nodes in fusible_groups.values():
        fusible_subset_list = []
        for subset in _get_independent_node_subsets(fusible_nodes, dependency_matrix):
            if len(subset) > 1:
                fusible_subset_list.append(subset)

        for subset in fusible_subset_list:
            fusion_rule.fuse(graph, subset)

            if isinstance(fusion_rule, GroupFusion):
                counters["inductor"]["group_fusion"] += 1
            else:
                counters["inductor"]["batch_fusion"] += 1


def group_batch_fusion_passes(graph: torch.fx.Graph):
    fusions = []

    if is_fbcode():
        fusions += [GroupLinearFusion()]

    for fusion_rule in fusions:
        apply_group_batch_fusion(graph, fusion_rule)
