import torch
import torch.nn as nn
from .policy import Policy_RNN_continous as Policy
from .quantized_ops import QuantizedConv2d_batch
import os


__all__ = ['MobileNetV1', 'mobilenet_v1']


Conv2d = QuantizedConv2d_batch


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv1 = conv_bn(3, 32, 2)
        self.model = nn.Sequential(
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)
        self.precision_selector = Policy(in_channels=32, num_layers=len(convs_list(self)))

    def forward(self, x):
        x = self.conv1(x)

        # Assign bitwidth to conv layers
        precision, log_probs, entropies = self.precision_selector(x)
        self.configure_model(precision, self.convs)

        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x, precision, log_probs, entropies

    def configure_model(self, bits, convs):
        bit_a, bit_w = bits
        for i, conv in enumerate(convs):
            conv.bitA = bit_a[:, i]
            conv.bitW = bit_w[:, i]

def convs_list(model):
    convs = list(filter(lambda x: isinstance(x, QuantizedConv2d_batch), [i for i in model.modules()]))
    return convs


def mobilenet_v1(pretrained=False, progress=True, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = torch.load('')
        model.load_state_dict(state_dict)
    return model
