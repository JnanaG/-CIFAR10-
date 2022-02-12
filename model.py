import torch.nn as nn
import torchsummary


class Lenet(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        # 输入为 227*227
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=20, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(156800, 500))  # 50*56*56=156800
        layer3.add_module('relu', nn.ReLU())
        layer3.add_module('fc2', nn.Linear(500, n_class))
        self.layer3 = layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.layer3(x)
        return output


class AlexNet(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(in_channels=in_dim, out_channels=96, kernel_size=11, stride=4, padding=0))
        layer1.add_module('bn1', nn.BatchNorm2d(96))
        layer1.add_module('relu', nn.ReLU())
        layer1.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer1 = layer1
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(9216, 4096),  # 256*6*6=9216
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

model = AlexNet(3, 10)
model.cuda()
torchsummary.summary(model, (3, 227, 227))