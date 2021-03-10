print("classifer.py")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import cv2

IMAGE_SIZE = 227

# Import dataset here and load using data-loader

# Perform transformations onthe dataset if needed
# confirm if the transformations should be preformed in Dataset class or independently

class GenderAgeClass(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
    def __len__(self): return len(self.df)
    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = f.file
        gen = f.gender == 'Female'
        age = f.age
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using: ", device)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv1 = nn.Conv2d(3, 96, 11, stride=1, padding=0)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        # Use local response normalization instead of batch normalization
        # self.bn1 = nn.BatchNorm2d(96, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        # Use local response normalization instead of batch normalization
        # self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.bn3 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.bn4 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu4 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(4096, 2)
        self.softmax = nn.Softmax(dim=None)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

print(Classifier)
mynet = Classifier().to(device)

loss_func = nn.MSELoss()
_Y = mynet(torch.rand(4,3,227,227))
Y = 1
loss_value = loss_func(_Y,Y)
print(loss_value)

from torch.optim import Adam
opt = Adam(mynet.parameters(), lr = 0.0001)    # As specified in the paper

loss_history = []
for _ in range(50):
    opt.zero_grad()
    loss_value = loss_func(mynet(X),Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value)

save_path = 'mymodel.pth'
torch.save(model.state_dict(), save_path)

load_path = 'mymodel.pth'
model.load_state_dict(torch.load(load_path))