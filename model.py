import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel
from efficientnet_pytorch import EfficientNet

class ResNet(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.network = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs,N)
        # self.network.fc = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(num_ftrs,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,N))
    
    def forward(self, x):
        out = self.network(x)
        return torch.sigmoid(out)
        

class Efficient_Net(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b4',num_classes = N)
    
    def forward(self, x):
        out = self.network(x)
        return torch.sigmoid(out)
        
 
class Mob_Net(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.network = models.mobilenet_v2(pretrained = True)
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(num_ftrs , N)
    
    def forward(self, x):
        out = self.network(x)
        return torch.sigmoid(out)
        
class ResNet152(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.network = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, N)
    
    def forward(self, x):
        out = self.network(x)
        return torch.sigmoid(out)
    
class VisionTransformer(nn.Module):
    def __init__(self,config,N):
        super().__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config)
        self.decoder = nn.Linear(768,N)

    def forward(self,x):
        encoder_out = self.encoder(x)
        out = self.decoder(encoder_out.pooler_output)
        return torch.sigmoid(out)
