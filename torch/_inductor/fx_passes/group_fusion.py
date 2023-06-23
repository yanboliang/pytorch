import collections
import operator

import torch

from ..pattern_matcher import (
    CallFunctionVarArgs,
)
from ..virtualized import V
from torch._dynamo.utils import counters

# import fbgemm_gpu  # Note: only after importing fbgemm_gpu, we can use `torch.ops.fbgemm.gmm`

aten = torch.ops.aten

def _get_dependent_nodes(node):
    input_nodes = node.all_input_nodes
    result_set = set(input_nodes)

    if input_nodes:
        for input_node in input_nodes:
            result_set.update(input_node.all_input_nodes)

    return result_set


MMNode = collections.namedtuple("MMNode", ['node', 'input', 'weight', 'bias'])

MMGroupKey = collections.namedtuple("MMGroupKey", ['m', 'k', 'n', 'has_bias'])


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
                dependency_matrix[i][j] = (
                    dependency_matrix[i][j] or
                    (dependency_matrix[i][k] and dependency_matrix[k][j])
                )

    return {
        nodes[i]: {
            nodes[j]: dependent
            for j, dependent in enumerate(row)
        }
        for i, row in enumerate(dependency_matrix)
    }


def _get_independent_node_subsets(mm_node_list, dependency_matrix):
    """
    Return an iterator of node subsets, each subset only contains nodes
    those are independent with each other.
    """

    def _mm_node_inputs(mm_node):
        nodes = [mm_node.input, mm_node.weight]
        if mm_node.bias is not None:
            nodes.append(mm_node.bias)
        return nodes

    def _is_independent(mm_node1, mm_node2):
        for n1 in _mm_node_inputs(mm_node1):
            for n2 in _mm_node_inputs(mm_node2):
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


def group_fusion_passes(graph: torch.fx.Graph):
    """
    fuse multiple torch.mm into one mm if possible
    """
    count = 0
    fusible_mm_group_dict = collections.defaultdict(list)

    dependency_matrix = _compute_graph_nodes_dependency_matrix(graph)

    for node in graph.nodes:
        if CallFunctionVarArgs(aten.mm.default).match(node):
            input_m, weight_m = node.args
            bias_m = None
        elif(
            CallFunctionVarArgs(aten.addmm.default).match(node) and
            node.kwargs.get("beta", 1.0) == 1.0 and
            node.kwargs.get("alpha", 1.0) == 1.0
        ):
            bias_m, input_m, weight_m = node.args
        else:
            continue

        m, k = input_m.meta["tensor_meta"].shape
        n = weight_m.meta["tensor_meta"].shape[1]

        group_key = MMGroupKey(m, k, n, bias_m is not None)

        fusible_mm_group_dict[group_key].append(MMNode(node, input_m, weight_m, bias_m))

    for group_key, fusible_mm_nodes in fusible_mm_group_dict.items():
        for mm_node_set in _get_independent_node_subsets(fusible_mm_nodes, dependency_matrix):
            if len(mm_node_set) <= 1:
                continue

            count += 1

            group_inputs = []
            group_weights = []
            group_biases = []
            group_nodes = []
            for node, input_m, weight_m, bias_m in mm_node_set:
                group_nodes.append(node)
                group_inputs.append(input_m)
                group_weights.append(weight_m)
                group_biases.append(bias_m)

            if all(bias is None for bias in group_biases):
                group_biases = None
            else:
                group_biases = [
                    0.0 if bias is None else bias
                    for bias in group_biases
                ]

            with graph.inserting_before(mm_node_set[0].node):
                fused_mm = graph.call_function(
                    torch.ops.fbgemm.gmm, args=(group_inputs, group_weights, group_biases)
                )

            for i, original_mm in enumerate(group_nodes):
                with graph.inserting_after(fused_mm):
                    new_mm = graph.call_function(
                        operator.getitem, args=(fused_mm, i)
                    )
                original_mm.replace_all_uses_with(new_mm)
                new_mm.meta.update(original_mm.meta)
                graph.erase_node(original_mm)
            
            counters["inductor"]["group_fusion"] += 1

    return count