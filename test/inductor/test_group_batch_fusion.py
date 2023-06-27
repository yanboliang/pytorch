# Owner(s): ["module: inductor"]

def gmm(*args, **kwargs):
    if len(args) == 2:
        inputs, weights = args
        biases=None
    else:
        inputs, weights, biases = args

    if biases is None:
        return tuple(
            torch.mm(input_m, weight_m)
            for input_m, weight_m in zip(inputs, weights)
        )
    return [
        torch.addmm(bias_m, input_m, weight_m)
        for input_m, weight_m, bias_m in zip(inputs, weights, biases)
    ]

gmm.__module__ = 'torch._ops.fbgemm'


import torch
torch.ops.fbgemm.gmm = gmm

from torch._dynamo.test_case import run_tests, TestCase
import torch._inductor
from torch._dynamo.utils import counters

class TestGroupFusion(TestCase):
    def test_group_linear_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, z: int, has_bias: bool) -> None:
                super().__init__()
                self.linear_w0 = torch.nn.Parameter(torch.randn(z, z))
                self.linear_w1 = torch.nn.Parameter(torch.randn(z, z))
                self.linear_w2 = torch.nn.Parameter(torch.randn(z, z))
                self.linear_w3 = torch.nn.Parameter(torch.randn(z, z))
                self.linear_w4 = torch.nn.Parameter(torch.randn(z, z))
                self.linear_w5 = torch.nn.Parameter(torch.randn(z, z))

                if has_bias:
                    self.linear_b0 = torch.nn.Parameter(torch.randn(z))
                    self.linear_b1 = torch.nn.Parameter(torch.randn(z))
                    self.linear_b2 = torch.nn.Parameter(torch.randn(z))
                    self.linear_b3 = torch.nn.Parameter(torch.randn(z))
                    self.linear_b4 = torch.nn.Parameter(torch.randn(z))
                    self.linear_b5 = torch.nn.Parameter(torch.randn(z))

            def forward(
                self,
                t0: torch.Tensor,
                t1: torch.Tensor,
                t2: torch.Tensor,
            ) -> torch.Tensor:
                a0 = torch.nn.functional.linear(
                    t0, weight=self.linear_w0, bias=self.linear_b0 if has_bias else None
                )
                a1 = torch.nn.functional.linear(
                    t1, weight=self.linear_w1, bias=self.linear_b1 if has_bias else None
                )
                a2 = torch.nn.functional.linear(
                    t2, weight=self.linear_w2, bias=self.linear_b2 if has_bias else None
                )

                b0 = torch.sigmoid(a0)
                b1 = torch.tanh(a1)
                b2 = torch.nn.functional.linear(
                    a2, weight=self.linear_w3, bias=self.linear_b3 if has_bias else None
                )

                c0 = b0 + b1 + b2
                c1 = torch.relu(b2)

                d0 = torch.nn.functional.linear(
                    c0, weight=self.linear_w4, bias=self.linear_b4 if has_bias else None
                )
                d1 = torch.nn.functional.linear(
                    c1, weight=self.linear_w5, bias=self.linear_b5 if has_bias else None
                )
                return d0 + d1

        Z = 16
        for has_bias in [True, False]:
            counters.clear()
            module = TestModule(Z, has_bias).eval()
            input = [
                torch.randn(4, Z),
                torch.randn(4, Z),
                torch.randn(4, Z),
            ]
            traced = torch.compile(module)
            self.assertTrue(torch.allclose(module(*input), traced(*input)))
            self.assertEqual(
                counters["inductor"]["group_fusion"],
                1,
            )
            counters.clear()


if __name__ == "__main__":
    run_tests()