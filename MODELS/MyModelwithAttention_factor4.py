import torch.nn as nn
import torch.nn.functional as F
import torch

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


class EdgeAugment1(nn.Module):
    def __init__(self, C, f1=3, f2=3, m1=3, m2=3):
        super(EdgeAugment1, self).__init__()
        self.conv1 = nn.Conv2d(1, C, kernel_size=f1, padding=f1 // 2)
        self.pool1 = nn.MaxPool2d(kernel_size=m1, stride=2, padding=m1//2)
        self.pool2 = nn.AvgPool2d(kernel_size=m2, stride=2, padding=m2//2)
        self.Denselinear1 = self._make_dense(C, C, 5)
        self.Denselinear2 = self.Dense(2 * C, 2 * C, 3)      #3
        self.Denselinear3 = self.Dense(C, C, 3)              #3
        self.upsample1 = nn.ConvTranspose2d(4 * C, C, kernel_size=3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(8 * C, C, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2 * C, 1, kernel_size=f2, padding=f2 // 2)
    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        layers.append(nn.Conv2d(in_channels=inChannels, out_channels=growthRate, kernel_size=1, padding=1 // 2))
        return nn.Sequential(*layers)
    def Dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
    def forward(self, x):
        f1 = self.conv1(x)            #b*C*41*41
        f2 = self.pool1(f1)           #b*C*21*21
        f3 = self.Denselinear1(f2)    #b*C*21*21
        f4 = self.pool2(f3)           #b*C*11*11
        f5 = self.Denselinear3(f4)    #b*4C*11*11
        f6 = self.upsample1(f5)       #b*C*21*21
        f7 = self.Denselinear2(torch.cat((f6, f3), 1))    #b*128*21*21
        f8 = self.upsample2(f7)       #b*16*41*41
        f9 = self.conv2(torch.cat((f8, f1), 1))    #b*1*41*41
        return F.sigmoid(f9), f1, f2, f3, f4, f5, f6, f7, f8


class Model(nn.Module):
    def __init__(self, growthRate=64, nDenselayer=3, k1=5, k2=3, f1=3, f2=3, m1=3, m2=3):
        super(Model, self).__init__()
        self.patch_extraction = nn.Conv2d(1, growthRate // 2, kernel_size=k1, padding=k1 // 2)
        self.linear = nn.Conv2d(growthRate // 2, growthRate, kernel_size=1, padding=1 // 2)

        inChannels = growthRate
        self.DenseConnection1 = self._make_dense(inChannels, growthRate, nDenselayer)
        self.DenseConnection2 = self._make_dense(inChannels, growthRate, nDenselayer)
        self.DenseConnection3 = self._make_dense(inChannels, growthRate, nDenselayer)

        self.upsampling = nn.ConvTranspose2d(in_channels=growthRate, out_channels=growthRate // 2, kernel_size=5, stride=4, padding=2)

        self.fusion_reconstruction = nn.Conv2d(growthRate // 2, 1, kernel_size=k2, padding=k2 // 2)

        self.EdgeDetection = EdgeAugment1(32, f1, f2, m1, m2)

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        layers.append(nn.Conv2d(in_channels=inChannels, out_channels=growthRate, kernel_size=1, padding=1 // 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out2 = F.interpolate(x, size=41, mode='bilinear')
        Detection, x1, x2, x3, x4, x5, x6, x7, x8 = self.EdgeDetection(out2)
        # Detection, _, _, _, _, _, _, _, _ = self.EdgeDetection(out2)
        output1 = F.relu(self.patch_extraction(x))
        output2 = F.relu(self.linear(output1))
        output3 = F.relu(self.DenseConnection1(output2))
        output4 = F.relu(self.DenseConnection2(output3))
        output5 = F.relu(self.DenseConnection3(output4 + output3))
        output6 = F.relu(self.upsampling(output5 + output2))
        out1 = self.fusion_reconstruction(output6)
        out = out1 * Detection + out2
        output7 = F.relu(out)
        return output7, Detection


