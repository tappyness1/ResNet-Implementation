import torch.nn as nn
from src.resnet_50_above import ResNet50Above
from src.resnet_34 import ResNet34
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

def train(epochs, train_set, in_channels = 3, num_classes = 10, save_model_path = "model_weights/model_weights.pt"):

    loss_function = nn.CrossEntropyLoss()
    # network = ResNet50Above(in_channels = in_channels, num_classes = num_classes)
    network = ResNet34(in_channels = in_channels, num_classes = num_classes)
    # summary(network, (in_channels,224,224))
    optimizer = optim.SGD(network.parameters(), lr=3e-4, weight_decay=5e-5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # print (device)
    network.to(device)
    # start = time.time()
    # add tqdm here
    train_dataloader = DataLoader(train_set, batch_size=20)
    for epoch in range(epochs):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:
                # print (img.shape)
                optimizer.zero_grad() 
                out = network.forward(imgs.to(device))
                loss = loss_function(out, labels)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
    print("training done")
    torch.save(network, save_model_path)

    return network

if __name__ == "__main__":

    # import numpy as np
    # import torch

    # np.random.seed(42)
    # torch.manual_seed(42)

    # X = np.random.rand(20, 3, 224, 224).astype('float32')
    # Y = torch.empty(20, dtype=torch.long).random_(2)
    # X = torch.tensor(X)
    # train(epochs = 10, X= X, Y=Y)

    torch.manual_seed(42)

    from src.dataset import get_load_data
    train_set, test_set = get_load_data(dataset = "Flowers102")
    train(epochs = 3, train_set = train_set, in_channels = 3, num_classes = 102)


    # train_set, test_set = get_load_data(dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    