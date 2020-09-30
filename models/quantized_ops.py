import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.distributed as dist


def PACT_A_batch(x, numBits, alpha):
    numBits = numBits.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    # x_clone = x.clone()

    # Clip to [0,alpha]
    w_q = 0.5*(torch.abs(x) - torch.abs(x-alpha) + alpha)

    # Quantize to k bits in range [0, alpha]
    w_q = quantize(w_q, numBits, alpha)

    # w_q[numBits == 32] = x_clone[numBits == 32]

    return w_q


def DoReFa_A_batch(x, numBits):
    numBits = numBits.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    # x_clone = x.clone()

    # DoReFa is the same than Half-wave Gaussian (uniform) Quantizer. They clip to [0,1]
    w_q = torch.clamp(x, min=0.0, max=1.0)

    # Quantize to k bits in range [0, 1]
    w_q = quantize(w_q, numBits)

    # w_q[numBits == 32] = x_clone[numBits == 32]

    return w_q


def PACT_A(x, numBits, alpha):
    if numBits == 32:
        return x

    # Clip to [0,alpha]
    w_q = 0.5*(torch.abs(x) - torch.abs(x-alpha) + alpha)

    # Quantize to k bits in range [0, 1]
    w_q = quantize(w_q, numBits, alpha)

    return w_q


def DoReFa_A(x, numBits):
    if numBits == 32:
        return x

    # DoReFa is the same than Half-wave Gaussian (uniform) Quantizer. They clip to [0,1]
    w_q = torch.clamp(x, min=0.0, max=1.0)

    # Quantize to k bits in range [0, 1]
    w_q = quantize(w_q, numBits)

    return w_q


def DoReFa_W(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    if numBits == 32:
        return x

    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    w_q = quantize(w_q, numBits)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


def quantize(x, k, alpha=1.0):
    n = (2**k.float() - 1.0)/alpha
    # n = (2**k - 1.0)/alpha           # Use for pretraining with [main_cheap | main_shared_step | main_step_per_switch]
    x = RoundNoGradient.apply(x, n)
    return x


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x*n)/n

    @staticmethod
    def backward(ctx, g):
        return g, None


class QuantizedConv2d_batch(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(QuantizedConv2d_batch, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias)

        self.clip = Parameter(torch.Tensor([1]))
        self.bitA = None
        self.bitW = None

    def forward(self, input):
        input = DoReFa_A_batch(input, self.bitA)

        weight = []
        for elem in torch.unique(self.bitW):
            weight.append(DoReFa_W(self.weight, elem))

        # This for loop runs in parallel in the gpu
        for i, elem in enumerate(torch.unique(self.bitW)):
            mask = self.bitW == elem
            _input = input[mask]
            res = F.conv2d(_input, weight[i], self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
            if 'output' not in locals():
                batch_size = input.size(0)
                size = (batch_size, self.out_channels, res.size(2), res.size(3))
                output = torch.zeros(size, dtype=res.dtype, device=res.device)
            output[mask] = res

        return output


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias)

        # self.clip = Parameter(torch.Tensor([1]))
        self.bitA = None
        self.bitW = None

    def forward(self, input):
        input = DoReFa_A(input, self.bitA)
        weight = DoReFa_W(self.weight, self.bitW)
        output = F.conv2d(input, weight, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)

        return output
