import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Lenet, AlexNex
from vgg16 import VGG16
import torchsummary
import warnings
warnings.filterwarnings('ignore')

# torch.cuda.empty_cache()
transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
# 超参数
batch_size = 64
lr = 1e-3
num_epochs = 10

# train_set = datasets.MNIST(root='/data/mnist',
#                            train=True,
#                            transform=transformer,
#                            download=True)
# val_set = datasets.MNIST(root='/data/mnist',
#                          train=False,
#                          transform=transformer)
train_set = datasets.CIFAR10(root='/data/cifar10',
                             train=True,
                             transform=transformer,
                             download=True)
val_set = datasets.CIFAR10(root='/data/cifar10',
                           train=False,
                           transform=transformer)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

model = torchvision.models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)# 采用这种方法，学习率以0.99指数衰减


def evaluate_acc(data_iter, model):
    correct = 0
    with torch.no_grad():
        for val_img, val_labels in data_iter:
            model.eval()
            val_img = val_img.to(device)
            val_labels = val_labels.to(device)
            output = model(val_img)
            _, predict = torch.max(output.data, dim=1)
            correct += (predict == val_labels).cpu().sum()
    return correct / len(data_iter)


for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)
    train_correct = 0
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicts = torch.max(outputs.data, dim=1)
        train_correct += torch.sum(predicts == labels)
        scheduler.step()  # 更新学习率
    epoch_loss = train_loss / len(train_loader)
    epoch_acc = train_correct / len(train_loader)
    print('Train Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    val_acc = evaluate_acc(val_loader, model)
    print("Valid acc: {}".format(val_acc))
