import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.fc1 = nn.Linear(24 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 50)
        self.out = nn.Linear(50, 8)

    def forward(self, t):
        t = self.conv1(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(t)

        t = self.conv2(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(t)

        t = self.conv3(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(t)

        t = t.reshape(-1, 24 * 6 * 6)

        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)

        return t

if __name__ == "__main__":
    import numpy as np
    import dataset_loader

    from PIL import Image

    conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
    conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)


    train_set, train_labels, test_set, test_labels = dataset_loader.load_h5_dataset()
    train_set, train_labels = dataset_loader.create_batch(train_set, train_labels, 32)

    image = train_set[0]

    t = conv1(image)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = F.relu(t)

    t = conv2(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = F.relu(t)

    t = conv3(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = F.relu(t)

    print(t.shape)