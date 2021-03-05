print('train.py Contaains the core training loop')

import torch
from torch import nn
import torchvision
import matplotlib
import numpy

class Classifier(nn.Module):
    def __init__(self):
        super.__init__():
        conv1 = nn.Conv2d(3, 96, 11, 4)
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(96, 256, 5, 2)
        pool2 = nn.MaxPool2d(2)
        conv3 = nn.Conv2d(256, 384, 3, 1)
        pool3 = nn.MaxPool2d(2)
        conv4 = nn.Conv2d(384, 256, 3, 1)
        pool4 = nn.MaxPool2d(2)
        fc1 = nn.fc(4096, 4096)
        fc2 = nn.fc(4096, 4096)
        fc3 = nn.fc(4096, 4096)
        bn = nn.BatchNorm2d(number out)
        soft = nn.Softmax()
        drop = nn.dropout(0.5)

    def forward():


for i in range(epochs):
    # Training loop


for j in range(len(val)):
    # Validation loop