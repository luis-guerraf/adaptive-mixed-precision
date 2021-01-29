import torch
import torch.nn as nn

switches = None


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(SwitchableBatchNorm2d, self).__init__()
        bns = []
        for _ in range(switches):
            bns.append(nn.BatchNorm2d(num_features))
        self.bn = nn.ModuleList(bns)
        self.switch = 0
        self.id = None

    def forward(self, input):
        y = self.bn[self.switch](input)
        return y


class SwitchableBatchNorm2d_batch(nn.Module):
    def __init__(self, num_features):
        super(SwitchableBatchNorm2d_batch, self).__init__()
        bns = []
        for _ in range(switches):
            bns.append(nn.BatchNorm2d(num_features))
        self.bn = nn.ModuleList(bns)
        self.switch = 0
        self.id = None

    def forward(self, input):
        output = torch.zeros(input.size(), dtype=input.dtype, device=input.device)

        # This for loop runs in parallel in the gpu
        for elem in torch.unique(self.switch):
            mask = self.switch == elem
            _input = input[mask]
            res = self.bn[elem](_input)
            output[mask] = res

        return output


def remap_BN(state_dict):
    temp_resnet = {}
    for k in state_dict.keys():
        temp_resnet[k] = state_dict[k]
        if 'layer' not in k:
            # First conv is not switchable
            continue
        if 'bn1' in k:
            temp_resnet[k.replace('bn1', 'bn1.bn.0')] = temp_resnet.pop(k)
        if 'bn2' in k:
            temp_resnet[k.replace('bn2', 'bn2.bn.0')] = temp_resnet.pop(k)
        if 'downsample.1' in k:
            temp_resnet[k.replace('downsample.1', 'downsample.1.bn.0')] = temp_resnet.pop(k)

    return temp_resnet


def replicate_SBN_params(model):
    for subm in model.modules():
        if (isinstance(subm, SwitchableBatchNorm2d)):
            for i, bn in enumerate(subm.bn):
                if i != 0:
                    bn.weight.data.copy_(subm.bn[0].weight.data)
                    bn.bias.data.copy_(subm.bn[0].bias.data)
                    bn.running_mean.data.copy_(subm.bn[0].running_mean.data)
                    bn.running_var.data.copy_(subm.bn[0].running_var.data)
                    bn.num_batches_tracked.data.copy_(subm.bn[0].num_batches_tracked.data)

