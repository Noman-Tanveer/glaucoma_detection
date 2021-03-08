print("classifer.py")

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Classifier(nn.Module):
    def __init__(self):
        super(self).__init__()
        #padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv1 = nn.Conv2d(3, 96, 11, stride, padding=padding)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(96, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.pool2 = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu2 = nn.ReLU(inplace=True) if relu else None

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.pool3 = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu3 = nn.ReLU(inplace=True) if relu else None

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.pool4 = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.bn4 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu4 = nn.ReLU(inplace=True) if relu else None

        self.fc1 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(4096, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x