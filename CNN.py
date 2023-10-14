"""
若是要在原函数上经过更改,在同样的程度上实现CNN,我的想法是将filter转化为一个向量,类似于一个移动的窗口与输入相乘得到结果
需要将向量转化为列表通过循环进行划分,经过变换后与filter相乘得到值,再将这些值组成一个新的向量,以此来达到类似filter的作用
在此之上,还需要多个filter进行维度转换
"""

"""
但我认为这道题应该是简单体验一下RRN吧(
以下仅仅是学习和试验内容
"""

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])
train_dataset = datasets.MNIST(root='/dataset/mnist/', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='/dataset/mnist/', train=False, download=True, transform=transforms)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=64)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.c1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.c2 = torch.nn.Conv2d(in_channels=20, out_channels=32, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(512, 206)
        self.fc2 = torch.nn.Linear(206, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.c1(x)))
        x = F.relu(self.pooling(self.c2(x)))
        x = x.view(batch_size, -1)
        x = self.fc2(self.fc1(x))
        return x
    
model = CNN()

criterion = torch.nn.CrossEntropyLoss()
ooptimizer = optim.SGD(model.parameters(), lr=0.1)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        ooptimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        ooptimizer.step()
        running_loss += loss.item()
        if batch_idx %300 == 299:
            print('[%d,%5d] loss:%.3f' %(epoch+1, batch_idx+1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print("正确率为： %d %%" % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            test()
