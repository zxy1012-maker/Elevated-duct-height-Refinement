import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import make_grid

class SingleLayer(nn.Module):
    def __init__(self, inChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv2d(inChannels, growthRate, kernel_size=3, padding=1, bias=True)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.drop(out)
        out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, growthRate=64, nDenselayer=3, k1=5, k2=3):
        super(Model, self).__init__()
        self.patch_extraction = nn.Conv2d(1, growthRate // 2, kernel_size=k1, padding=k1 // 2)
        self.linear = nn.Conv2d(growthRate // 2, growthRate, kernel_size=1, padding=1 // 2)

        inChannels = growthRate
        self.DenseConnection1 = self._make_dense(inChannels, growthRate, nDenselayer)
        self.DenseConnection2 = self._make_dense(inChannels, growthRate, nDenselayer)
        self.DenseConnection3 = self._make_dense(inChannels, growthRate, nDenselayer)

        self.upsampling = nn.ConvTranspose2d(in_channels=growthRate, out_channels=growthRate // 2, kernel_size=3, stride=2, padding=1)

        self.fusion_reconstruction = nn.Conv2d(growthRate // 2, 1, kernel_size=k2, padding=k2 // 2)

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        layers.append(nn.Conv2d(in_channels=inChannels, out_channels=growthRate, kernel_size=1, padding=1 // 2))
        return nn.Sequential(*layers)
    def forward(self, x):
        output1 = F.relu(self.patch_extraction(x))
        output2 = F.relu(self.linear(output1))
        output3 = F.relu(self.DenseConnection1(output2))
        output4 = F.relu(self.DenseConnection2(output3))
        output5 = F.relu(self.DenseConnection3(output4 + output3))
        output6 = F.relu(self.upsampling(output5 + output2))
        out1 = self.fusion_reconstruction(output6)
        out2 = F.interpolate(x, size=21, mode='bilinear')
        out = out1 + out2
        output7 = F.relu(out)
        return output7, x


