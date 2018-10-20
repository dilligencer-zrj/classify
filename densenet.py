#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))  # BN,RELU, conv:1*1
        out = self.conv2(F.relu(self.bn2(out))) # BN,RELU, conv:3*3
        out = torch.cat((x, out), 1) # 拼接
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) # BN,RELU, conv:3*3
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) # BN,RELU, conv:1*1
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_1(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet_1, self).__init__()

        nDenseBlocks = (depth-4) // 3  # 32
        if bottleneck:
            nDenseBlocks //= 4  # 16

        nChannels = 2*growthRate  # 24
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck) # 16层
        nChannels += nDenseBlocks*growthRate  #216
        nOutChannels = int(math.floor(nChannels*reduction)) # 108
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels # 108
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck) # 16层
        nChannels += nDenseBlocks*growthRate # 300
        nOutChannels = int(math.floor(nChannels*reduction)) # 150
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck) # 16层
        nChannels += nDenseBlocks*growthRate  #342
        nOutChannels = int(math.floor(nChannels * reduction)) # 171
        self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)  # 16层
        nChannels += nDenseBlocks * growthRate  # 363
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans4 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels #181
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)  # 16层
        nChannels += nDenseBlocks * growthRate  # 373

        self.bn1 = nn.BatchNorm2d(nChannels)

        self.fc1 = nn.Linear(nChannels*16*16, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.trans4(self.dense4(out))
        out = self.dense5(out)
        out = F.avg_pool2d(F.relu(self.bn1(out)), 2)
        out=out.view(out.size(0),-1)
        out = F.log_softmax(self.fc1(out))
        return out

if __name__ == '__main__':
    densenet = DenseNet_1(growthRate=12, depth=60, reduction=0.5,
                        bottleneck=True, nClasses=14)
    densenet.cuda()