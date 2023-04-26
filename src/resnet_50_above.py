from src.resnet_utils import IdentityBlock
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
from src.resnet_utils import IdentityBlock, forward_layer
import torch.nn as nn

class ResNet50Above(nn.Module):

    def __init__(self, num_classes, variant = "ResNet50"):
        
        super(ResNet50Above, self).__init__()
        self.relu = ReLU()

        if variant == "ResNet50":
            layers = [2,3,5,2]

        elif variant == "ResNet101":
            layers = [2,3,22,2]
        
        elif variant == "ResNet152":
            layers = [2,7,35,2]

        else:
            raise Exception("Choose either 'ResNet50', 'ResNet101' or 'ResNet152'.")

        # conv1
        self.conv1 = Conv2d(in_channels = 3, out_channels=64, kernel_size=7, stride = 2)
        self.conv1_bn = BatchNorm2d(num_features=64)

        # conv2_x
        self.maxpool1 = MaxPool2d(kernel_size=3, stride = 2)
        self.id2_1 = IdentityBlock(in_channels = 64, out_channels = [64, 64, 256], conv = False)
        self.layer2 = [IdentityBlock(in_channels = 256, out_channels = [64, 64, 256], conv = False) for i in range(layers[0])]
        
        # conv3_x
        self.id3_1 = IdentityBlock(in_channels = 256, out_channels = [128, 128, 512], conv = True)
        self.layer3 = [IdentityBlock(in_channels = 512, out_channels = [128, 128, 512], conv = False) for i in range(layers[1])]

        # conv4_x
        self.id4_1 = IdentityBlock(in_channels = 512, out_channels = [256, 256, 1024], conv = True)
        self.layer4 = [IdentityBlock(in_channels = 1024, out_channels = [256, 256, 1024], conv = False) for i in range(layers[2])]
        
        # conv5_x
        self.id5_1 = IdentityBlock(in_channels = 1024, out_channels = [512, 512, 2048], conv = True)
        self.layer5 = [IdentityBlock(in_channels = 2048, out_channels = [512, 512, 2048], conv = False) for i in range(layers[3])]

        # avg pool, 1000 fc and softmax
        self.avg_pool = AdaptiveAvgPool2d((1,1)) # pytorch's implementation uses adaptive. Not sure what's the diff
        self.fc = Linear(2048,num_classes) # 1000 dim fc
        self.softmax = Softmax(dim = 1)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv1_bn(out)
        out = self.relu(out)

        out = self.maxpool1(out)
        out = self.id2_1(out)
        out = forward_layer(out, self.layer2)

        out = self.id3_1(out)
        out = forward_layer(out, self.layer3)
        
        out = self.id4_1(out)
        out = forward_layer(out, self.layer4)

        out = self.id5_1(out)
        out = forward_layer(out, self.layer5)

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

    resnet_101 = ResNet50Above(variant = 'ResNet101', num_classes= 3)
    print (resnet_101.forward(X))