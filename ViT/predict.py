from timm.models import create_model
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import DrowsinessDataset
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import torch

class FastViTModel(nn.Module):
    def __init__(self):
        super(FastViTModel,self).__init__()
        self.fastvit=create_model("fastvit_t8")
        '''torch.save({
            'state_dict': self.fastvit.state_dict(),
            
        }, 'fastvit_t8.pth.tar')
        checkpoint = torch.load('fastvit_t8.pth.tar')
        self.fastvit.load_state_dict(checkpoint['state_dict'])
        self.fastvit.eval()'''
        
    def forward(self,x):
        with torch.no_grad():
            features=self.fastvit(x)
        return features
class CombinedModel(nn.Module):
    def __init__(self,embedding_size,hidden_layer,output_layer):
        super(CombinedModel,self).__init__()
        self.fastvitmodel=FastViTModel()
        self.dropout=nn.Dropout(p=0)
        self.fc1=nn.Linear(embedding_size+1,hidden_layer)
        self.fc2=nn.Linear(hidden_layer,output_layer)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,image,ear):
        img_ten=self.fastvitmodel(image)
        ear = ear.view(-1, 1)
        combined_ten=torch.cat([img_ten,ear],dim=1)
        x=torch.relu(self.fc1(combined_ten))
        x=self.dropout(x)
        output=self.sigmoid(self.fc2(x))
        return output

embedd=1000
hidden=720
output_feats=1
model=CombinedModel(embedd,hidden,output_feats)
ckpt=torch.load('best_loss.pth',map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
print("Model loaded at epoch:",ckpt['epoch']," at loss: ",ckpt['loss'])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test=pd.read_csv('test_annotations.csv')
test_dataset=DrowsinessDataset(test,transform=transform)
test_dl = DataLoader(test_dataset, batch_size=36, shuffle=True)

all_labels = []
all_predictions = []
test_loss = 0
correct = 0
model.eval()
device='cpu'
criterion = torch.nn.BCEWithLogitsLoss()
with torch.no_grad():  
    for image, ear, label in tqdm(test_dl, desc="Test"): 
        image=image.to(device)
        ear=ear.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        output = model(image,ear)
        output=output.squeeze(1)  
        test_loss += criterion(output, label).item()  
        pred = (output > 0.5).float() 
        correct += pred.eq(label.view_as(pred)).sum().item()
        all_predictions.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

test_loss /= len(test_dl.dataset)
accuracy = 100. * correct / len(test_dl.dataset)
precision = precision_score(all_labels, all_predictions, average="binary")
recall = recall_score(all_labels, all_predictions, average="binary")
f1 = f1_score(all_labels, all_predictions, average="binary")

print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dl.dataset)} ({accuracy:.2f}%)\n')
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")