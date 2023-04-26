from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential
import numpy as np
import torch.nn as nn
import torch

def forward_layer(out, conv_list):
    for conv in conv_list:    
        out = conv(out)
    return out

class IdentityBlock(nn.Module):
        
    def __init__(self, in_channels= 3, out_channels = [64, 64], conv = False):
        super().__init__()

        """generate identity block for resnet
        conv == True -> conv1 will have stride = 2, kernel_size = 1 
        conv == False -> conv1 will have stride = 1, kernel_size = 3, padding is same
        """
        assert len(out_channels) in [2,3], "Can only have 2 or 3 depth in id block"
        if len(out_channels) == 3:
            self.bottleneck = True
        else: self.bottleneck = False

        self.last_filter_channel = out_channels[-1]
        self.conv = conv

        if conv:

            self.conv1 = Conv2d(in_channels=in_channels, out_channels = out_channels[0], kernel_size = 1, stride = 2, padding='valid', bias = False)
            # if our kernel_size != 1, then padding has to be set to "same" because you don't have to deal with dimension change issues
        elif self.bottleneck:
            self.conv1 = Conv2d(in_channels=in_channels, out_channels = out_channels[0], kernel_size = 1, stride = 1, padding='same', bias = False)
       
        else: 
            self.conv1 = Conv2d(in_channels=in_channels, out_channels = out_channels[0], kernel_size = 3, stride = 1, padding='same', bias = False)
        
        self.batchnorm1 = BatchNorm2d(num_features=out_channels[0])

        self.conv2 = Conv2d(in_channels= out_channels[0], out_channels = out_channels[1], kernel_size= 3, stride = 1, padding = 'same', bias = False)
        self.batchnorm2 = BatchNorm2d(num_features=out_channels[1])

        if self.bottleneck:
            self.conv3 = Conv2d(in_channels= out_channels[1], out_channels = out_channels[2], kernel_size= 1, stride = 1, padding = 'same', bias = False)
            self.batchnorm3 = BatchNorm2d(num_features=out_channels[2])

        self.relu = ReLU()

    def forward(self, X):
        
        identity = X
        # to do - downsample when we expect the output channel here to be more than the identity
        # emply 1x1 conv layer with batchnorm

        if identity.shape[1] != self.last_filter_channel:
            if self.conv:
                conv_id = Conv2d(identity.shape[1], self.last_filter_channel, kernel_size=1, stride = 2, padding ='valid')
            else:
                conv_id = Conv2d(identity.shape[1], self.last_filter_channel, kernel_size=1, padding = 'same')
            
            batchnorm = BatchNorm2d(num_features=self.last_filter_channel)
            
            identity = conv_id(identity)
            identity = batchnorm(identity)

        out = self.conv1(X)
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)

        if self.bottleneck: 
            out = self.relu(out)
            out = self.conv3(out)
            out = self.batchnorm3(out) 

        out += identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    import tests.resnet_utils_test as resnet_test

    resnet_test.test_id_conv_true_same_channel()
    resnet_test.test_id_conv_true_diff_channels()
    resnet_test.test_id_conv_false_diff_channels()
    resnet_test.test_id_block()

    print ("All passed")