from src.resnet_utils import IdentityBlock
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
from src.resnet_utils import IdentityBlock, forward_layer
import torch.nn as nn
import torch

class ResNet34(nn.Module):

    def __init__(self, num_classes, in_channels = 3):
        super(ResNet34, self).__init__()
        self.relu = ReLU()

        # conv1
        self.conv1 = Conv2d(in_channels = in_channels, out_channels=64, kernel_size=7, stride = 2)
        self.conv1_bn = BatchNorm2d(num_features=64)

        # conv2_x
        self.maxpool1 = MaxPool2d(kernel_size=3, stride = 2)
        self.id2_1 = IdentityBlock(in_channels = 64, out_channels = [64, 64], conv = False)
        self.id2_2 = IdentityBlock(in_channels = 64, out_channels = [64, 64], conv = False)
        self.id2_3 = IdentityBlock(in_channels = 64, out_channels = [64, 64], conv = False)
        
        # conv3_x
        self.id3_1 = IdentityBlock(in_channels = 64, out_channels = [128, 128], conv = True)
        self.id3_2 = IdentityBlock(in_channels = 128, out_channels = [128, 128], conv = False)
        self.id3_3 = IdentityBlock(in_channels = 128, out_channels = [128, 128], conv = False)
        self.id3_4 = IdentityBlock(in_channels = 128, out_channels = [128, 128], conv = False)

        # conv4_x
        self.id4_1 = IdentityBlock(in_channels = 128, out_channels = [256, 256], conv = True)
        self.id4_2 = IdentityBlock(in_channels = 256, out_channels = [256, 256], conv = False)
        self.id4_3 = IdentityBlock(in_channels = 256, out_channels = [256, 256], conv = False)
        self.id4_4 = IdentityBlock(in_channels = 256, out_channels = [256, 256], conv = False)
        self.id4_5 = IdentityBlock(in_channels = 256, out_channels = [256, 256], conv = False)
        self.id4_6 = IdentityBlock(in_channels = 256, out_channels = [256, 256], conv = False)

        # conv5_x
        self.id5_1 = IdentityBlock(in_channels = 256, out_channels = [512, 512], conv = True)
        self.id5_2 = IdentityBlock(in_channels = 512, out_channels = [512, 512], conv = False)
        self.id5_3 = IdentityBlock(in_channels = 512, out_channels = [512, 512], conv = False)

        # avg pool, 1000 fc and softmax
        self.avg_pool = AdaptiveAvgPool2d((1,1)) # pytorch's implementation uses adaptive. Not sure what's the diff
        self.fc = Linear(512,num_classes) # 1000 dim fc
        self.softmax = Softmax(dim = 1)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv1_bn(out)
        out = self.relu(out)

        out = self.maxpool1(out)
        out = self.id2_1(out)
        out = self.id2_2(out)
        out = self.id2_3(out)

        out = self.id3_1(out)
        out = self.id3_2(out)
        out = self.id3_3(out)
        out = self.id3_4(out)
        
        out = self.id4_1(out)
        out = self.id4_2(out)
        out = self.id4_3(out)
        out = self.id4_4(out)
        out = self.id4_5(out)
        out = self.id4_6(out)

        out = self.id5_1(out)
        out = self.id5_2(out)
        out = self.id5_3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        out = self.softmax(out)

        return out

if __name__ == "__main__":
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 32, 32).astype('float32')

    X = torch.tensor(X)

    resnet_34 = ResNet34(num_classes= 3, variant = "ResNet34")
    print (resnet_34.forward(X))