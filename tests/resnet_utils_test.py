from src.resnet_utils import IdentityBlock
import numpy as np
import torch

def test_id_conv_true_same_channel():
    
    np.random.seed(1)
    X = np.random.rand(3, 5, 25, 20).astype('float32')
    X = torch.tensor(X)
    id_block = IdentityBlock(X.shape[1], out_channels = [64, 64], conv = True)

    assert id_block.forward(X).shape == torch.Size([3, 64, 13, 10]), "Not Right"

def test_id_conv_true_diff_channels():
    np.random.seed(1)
    X = np.random.rand(3, 5, 25, 20).astype('float32')
    X = torch.tensor(X)
    id_block = IdentityBlock(X.shape[1], out_channels = [64, 128], conv = True)
    assert id_block.forward(X).shape == torch.Size([3, 128, 13, 10]), "Not Right"

def test_id_conv_false_diff_channels():
    np.random.seed(1)
    X = np.random.rand(3, 5, 25, 20).astype('float32')
    X = torch.tensor(X)
    id_block = IdentityBlock(X.shape[1], out_channels = [64, 128], conv = False)
    assert id_block.forward(X).shape == torch.Size([3, 128, 25, 20]), "Not Right"

def test_id_block():
    np.random.seed(1)
    X = np.random.rand(3, 5, 25, 20).astype('float32')
    X = torch.tensor(X)
    id_block = IdentityBlock(X.shape[1], out_channels = [64, 64], conv = False)
    assert id_block.forward(X).shape == torch.Size([3, 64, 25, 20]), "Not Right"