import torch
from torch import nn
from torch.nn import functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, widen_factor=4):
        super(Bottleneck, self).__init__()
        D = out_channel // widen_factor
        self.conv_reduce = nn.Conv2d(in_channel, D, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, 3, stride=stride, padding=1, bias=False, groups=32)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channel, 1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channel)
        self.shortcut = None

        if in_channel!=out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, input):
        x = F.relu(self.bn_reduce(self.conv_reduce(input)), inplace=True)
        x = F.relu(self.bn(self.conv_conv(x)), inplace=True)
        x = F.relu(self.bn_expand(self.conv_expand(x)), inplace=True)
        resdual = self.shortcut(input) if self.shortcut else input
        return F.relu(resdual+x, inplace=True)

class ResneXt(nn.Module):
    def __init__(self, nc):
        super(ResneXt, self).__init__()
        blocks = [3, 4, 6, 3]
        channels = [256, 512, 1024, 2048]

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = self.make_blocks(64, channels[0], 1, blocks[0], 2)
        self.layer2 = self.make_blocks(channels[0], channels[1], 2, blocks[1], 2)
        self.layer3 = self.make_blocks(channels[1], channels[2], 2, blocks[2], 2)
        # self.layer4 = self.make_blocks(channels[2], channels[3], 2, blocks[3])

        self.fc = nn.Linear(1024, nc)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = F.adaptive_max_pool2d(x, (1,1))
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    def make_blocks(self, in_channel, out_channel, stride, depth, widen_factor=4):
        stage = []
        for i in range(depth):
            if not i:
                stage.append(Bottleneck(in_channel, out_channel, stride, widen_factor))
            else:
                stage.append(Bottleneck(out_channel, out_channel, 1, widen_factor))

        return nn.Sequential(*stage)