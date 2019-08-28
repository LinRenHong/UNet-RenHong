
### ------------------------------------
### This Network can be modify structure
### ------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

from models.unet import UNet
from models.unet.unet_parts import *
from config import config

opt = config

class WoundUNet(UNet):
    def __init__(self, n_channels, n_classes):
        super(WoundUNet, self).__init__(n_channels=n_channels, n_classes=n_classes)

        test = list(enumerate(self.children()))
        print("Layers amount: {}".format(len(test)))

        self.down3 = down(256, 256)

    def forward(self, x):
        print("Origin: {}".format(x.shape))
        x1 = self.inc(x)
        print("After inc: {}".format(x1.shape))
        x2 = self.down1(x1)
        print("After down1: {}".format(x2.shape))
        x3 = self.down2(x2)
        print("After down2: {}".format(x3.shape))
        x4 = self.down3(x3)
        print("After down3: {}".format(x4.shape))
        # x5 = self.down4(x4)
        # print("After down4: {}".format(x5.shape))

        # x = self.up1(x5, x4)
        # print("After up1: {}".format(x.shape))
        x = self.up2(x4, x3)
        print("After up2: {}".format(x.shape))
        x = self.up3(x, x2)
        print("After up3: {}".format(x.shape))
        x = self.up4(x, x1)
        print("After up4: {}".format(x.shape))

        x = self.outc(x)
        print("After outc: {}".format(x.shape))

        return torch.sigmoid(x)
