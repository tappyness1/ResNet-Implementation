import torch.nn as nn
from src.resnet_50_above import ResNet50Above
from src.resnet_34 import ResNet34
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd

def validation(model, val_set):
    model.eval()
    val_dataloader = DataLoader(val_set, batch_size=20)
    loss_function = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []

    with tqdm(val_dataloader) as tepoch:
        for imgs, labels in tepoch:
            y_true.extend(labels.numpy())
            with torch.no_grad():
                out = model(imgs)
            y_pred.extend(torch.argmax(out,1).cpu().numpy())
            loss = loss_function(out, labels)    

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cm.to_csv("val_results/results.csv")
    print (cm)

if __name__ == "__main__":
    
    from src.dataset import get_load_data

    _, val_set = get_load_data(dataset = "Flowers102")
    trained_model_path = "model_weights/model_weights.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))
    validation(model, val_set)
            