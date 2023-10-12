import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 2))
        self.pool = nn.MaxPool2d(1, 2)
        self.conv2 = nn.Conv2d(32, 16, (2, 1))
        self.fc1 = nn.Linear(16 * 2 * 1, 1)

    def forward(self, x):
        # print("inside cnn, shape: ", x.shape)
        batch_size = x.size(0)
        x = x.reshape(batch_size, 1, 12, 2)
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x
