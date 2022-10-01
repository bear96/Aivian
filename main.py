import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, help="Directory of the dataset for training.")
parser.add_argument("--valid_dir", type=str, help="Directory of the dataset for validation.")
parser.add_argument("--test_dir", type=str, help="Directory of the dataset for testing.")
parser.add_argument("--save_dir", default = 'resnet_birds.pkl',type=str, help="Directory of the dataset for saving model.")
parser.add_argument("--batch_size", default = 128, type = int, help = "Batch size for training and validation.")
parser.add_argument("--epochs", default = 10, type = int, help = "Number of epochs for training.")
parser.add_argument("--lr",default = 0.0005, type = float, help = "Learning rate of optimizer.")
parser.add_argument("--augment", action ='store_true',default= False, help = "Use if augmenting dataset.")
args = parser.parse_args()
print(args)


train_dir  = args.train_dir
valid_dir = args.valid_dir
test_dir = args.test_dir
classes = os.listdir(train_dir)

transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = ImageFolder(train_dir, transform = transformations)
valid_dataset = ImageFolder(valid_dir, transform = transformations)
test_dataset = ImageFolder(test_dir, transform = transformations)

random_seed = 30
torch.manual_seed(random_seed)
batch_size = args.batch_size

if args.augment:
  data_aug = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomRotation(degrees=10), 
                               transforms.ColorJitter(brightness=0.5,hue=0.3),transforms.GaussianBlur(kernel_size=11),
                                 transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  augmented_train_dataset = ImageFolder(train_dir,transform=data_aug)
  new_dataset = torch.utils.data.ConcatDataset([train_dataset,augmented_train_dataset])
else:
  new_dataset = train_dataset

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device: ",device)

model = model.ResNet().to(device)

num_epochs = args.epochs
optimizer = torch.optim.Adam(model.parameters(),lr= args.lr)
criterion = nn.CrossEntropyLoss()

train_dl = DataLoader(new_dataset, batch_size, shuffle = True, num_workers = 1)
val_dl = DataLoader(valid_dataset, batch_size, num_workers = 1)

for i in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for data, label in tqdm(train_dl):
        data = data.to(device)
        label = label.to(device)
        out_probs = model(data)
        loss = criterion(out_probs,label)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        
    model.val()
    for data, label in tqdm(val_dl):
      data = data.to(device)
      label = label.to(device)
      out_probs = model(data)
      loss = criterion(out_probs,label)
      val_los += loss.item()
   
  print("Epoch: {} Train loss: {} Val Loss: {}".format(i+1,train_loss/len(train_dl),val_loss/len(val_dl))
  torch.save(model.state_dict(), args.save_dir)
    
