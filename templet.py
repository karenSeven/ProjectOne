import time
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import torch.nn.functional as F

start = time.process_time()

# constant
DOWNLOAD_MNIST = False
EPOCH = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOG_INTERVAL = 100
train_loss = []
train_acc = []

train_data = torchvision.datasets.MNIST('./mnist_data', train=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
                                        download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST('./mnist_data/', train=False, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

train_loader1 = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader1 = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

print(train_data.data.shape)  # torch.Size([60000, 28, 28])
print(test_data.targets[:10], test_data.targets.shape)  # torch.Size([10000, 28, 28])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, ),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2), )  # (64, 1, 32, 32) -> (64, 16, 28, 28) -> (64, 16, 14, 14)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), ) # (batch, 32, 7, 7) -> (batch, 10)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #(batch, 32*7*7)
        output = self.out(x)
        return output


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE=28):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=2,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.2,
            bidirectional=True
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, _ = self.rnn(x, None)  # None represents zero initial hidden state
        # (h_n, h_c)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()   # 64, 784
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),  # 64, 512
            nn.ReLU(),
            nn.Linear(512, 256),  # 64, 256
            nn.ReLU(),
            nn.Linear(256, 128)   # 64, 128
        )
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        output = self.output(x)
        return output


def train(train_loader, model, optimizer, loss_func, epoch):
    model.train()
    for step, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # data = data.reshape(data.shape[0], -1)
        # data = data.view(-1, 28, 28)  # RNN

        output = model(data)
        loss = loss_func(output, target)
        # print(type(output), output.shape, type(target), target.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset), loss.item()))
            train_loss.append(loss.item())
            train_acc.append(metrics.accuracy_score(target.cpu(), torch.max(output.cpu(), 1)[1].data))


def test(model, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # data = data.reshape(data.shape[0], -1)
            # data = data.view(-1, 28, 28)  # RNN

            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def drawGroundTruth():
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.show()


def main():
    model = CNN().cuda()
    save_load = False
    if save_load:
        model.load_state_dict(torch.load('mnist_cnn.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # optimize all parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    test(model, test_loader1, loss_func)
    for epoch in range(1, EPOCH + 1):
        train(train_loader1, model, optimizer, loss_func, epoch)
        test(model, test_loader1, loss_func)

    save_model = True
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def visual():
    x = np.arange(1, train_loss.size + 1) * 10
    plt.title("Matplotlib demo")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    # plt.plot(x, train_loss)
    plt.plot(x, train_acc)
    plt.show()


# drawGroundTruth()
main()

train_loss = np.array(train_loss)
train_acc = np.array(train_acc)
visual()

elapsed = (time.process_time() - start)
print("Time used:", elapsed)
