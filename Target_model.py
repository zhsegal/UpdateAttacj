import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ShadowModel(nn.Module):
    def __init__(self, data_set, in_channels, out_channels, fc_size):
        super(ShadowModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels[0], out_channels=out_channels[1], kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        if data_set == "Mnist":
            self.fc = nn.Sequential(
                nn.Linear(fc_size, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
            )

        elif data_set == "Cifar":
            self.fc = nn.Sequential(
                nn.Linear(fc_size, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84,10)
            )

    def forward(self, sample):
        out = self.conv(sample)
        out = self.fc(out.view(sample.size(0), -1))
        out = torch.softmax(out, dim=1)
        return out


# do one epoch on the update set and return the update model
def update_model(net, set):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    update_loader = torch.utils.data.DataLoader(set, batch_size=64,
                                               shuffle=True, num_workers=2)

    net.train()
    for epoch in range(10):
        for i, (images, labels) in enumerate(update_loader):
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    return net


# train the shadow model
def train_shadow(train_set, data_set, transform, num_epochs=100, lr=0.1, batch_size=64):

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)

    if data_set == "Mnist":
        set = torchvision.datasets.MNIST
        in_channels = 1
        out_channels = [10, 20]
        fc_size = 320   
    elif data_set == "Cifar":
        set = torchvision.datasets.CIFAR10
        in_channels = train_set.data.shape[3]
        out_channels=[6,16]
        fc_size = 400
    else:
        raise NotImplementedError
    
    test_set = set(root='./data', train=False,
                                           download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=200,
                                             shuffle=False, num_workers=2)

        
    # batch_size, epoch and iteration
    model = ShadowModel(data_set, in_channels, out_channels, fc_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)


    loss_list = []
    accuracy_list = []
    t0 = time.time()
    print("Starting training...\n")
    # iterate over train set:
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

        scheduler.step()

        correct = 0
        total = 0

    # Iterate through valid dataset for accuracy
        model.eval()
        for images, labels in test_loader:
            outputs = model(images.to(device))
            valid_loss = criterion(outputs, labels.to(device))
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
        accuracy = 100 * correct / float(total)
        # store loss and iteration
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

        print('Epoch: {}  Loss: {:.3f}  Accuracy: {:.3f} %'.format(epoch, loss.data.item(), accuracy))

    return model, accuracy_list


def plot(accuracy_list):
    plt.title("target model accuracy")
    plt.plot(range(len(accuracy_list)), accuracy_list, label="test")
    plt.legend()
    plt.show()
    plt.savefig("target_model_acuuracy.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)    # learning rate
    parser.add_argument('--epoch', type=int, default=50)   # number of epochs for training
    parser.add_argument('--batch_size', type=int, default=64)   # number of epochs for training
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    model, accuracy_list = train_shadow(train_set, num_epochs=args.epoch, lr=args.lr, batch_size=args.batch_size)

    torch.save(model, 'saved/model_shadow.pth')
