# Owner(s): ["module: inductor"]


def group_gemm(*args, **kwargs):
    if len(args) == 2:
        inputs, weights = args
        biases = None
    else:
        inputs, weights, biases = args

    if biases is None:
        return tuple(
            torch.mm(input_m, weight_m) for input_m, weight_m in zip(inputs, weights)
        )
    return [
        torch.addmm(bias_m, input_m, weight_m)
        for input_m, weight_m, bias_m in zip(inputs, weights, biases)
    ]


group_gemm.__module__ = "torch._ops.fbgemm"


import torch

torch.ops.fbgemm.group_gemm = group_gemm

import torch._inductor
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters


class MyModule(torch.nn.Module):
    def __init__(self, z: int, has_bias: bool) -> None:
        super().__init__()
        self.linear0 = torch.nn.Linear(z, z, has_bias)
        self.linear1 = torch.nn.Linear(z, z, has_bias)
        self.linear2 = torch.nn.Linear(z, z, has_bias)
        self.linear3 = torch.nn.Linear(z, z, has_bias)
        self.linear4 = torch.nn.Linear(z, z, has_bias)
        self.linear5 = torch.nn.Linear(z, z, has_bias)
        self.bn0 = torch.nn.BatchNorm1d(z)
        self.bn1 = torch.nn.BatchNorm1d(z)
        self.bn2 = torch.nn.BatchNorm1d(z)
        self.bn3 = torch.nn.BatchNorm1d(z)
        self.bn4 = torch.nn.BatchNorm1d(z)

    def forward(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        a0 = self.bn0(self.linear0(t0))
        a1 = self.bn1(self.linear1(t1))
        a2 = self.bn2(self.linear2(t2))

        b0 = torch.sigmoid(a0)
        b1 = torch.tanh(a1)
        b2 = self.linear3(a2)

        c0 = b0 + b1 + b2
        c1 = torch.relu(b2)

        d0 = self.bn3(self.linear4(c0))
        d1 = self.bn4(self.linear5(c1))
        return d0 + d1


class TestGroupFusion(TestCase):
    def test_group_linear_fusion(self):
        z = 16
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule(z, has_bias).eval()
            input = [
                torch.randn(4, z),
                torch.randn(4, z),
                torch.randn(4, z),
            ]
            traced = torch.compile(module)
            self.assertTrue(torch.allclose(module(*input), traced(*input)))
            self.assertEqual(
                counters["inductor"]["group_fusion"],
                2,
            )
            counters.clear()


if __name__ == "__main__":
    run_tests()
