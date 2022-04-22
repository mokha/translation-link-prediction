import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from common import *
from sklearn.model_selection import train_test_split
import datetime, time, copy
import math
import random
from torch.utils.data import DataLoader, Dataset
# from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(4, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        # self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)
        # self.dropout3 = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        # x = self.dropout1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        # x = self.dropout2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        # x = self.dropout3(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def main():
    dataX, dataY = wiki_binary_predictions()
    print(len(dataX), len(dataY))
    X, y = [], []
    for k in dataX.keys():
        dx = dataX[k]
        if len(dx) != 4:
            continue
        X.append(dx)
        y.append([dataY[k]])
    print(len(X), len(y))

    test_data = annotated_predictions()
    test_data = test_data.values()
    test_data = [(d[1:], [d[0], ]) for d in test_data]
    X_test = [d[0] for d in test_data]
    y_test = [d[1] for d in test_data]
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.20, random_state=42)

    X = X_train  # X + X_train
    y = y_train  # y + y_train

    model = binaryClassification()
    model.to(device)

    EPOCHS = 1000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # train_data = trainData(torch.FloatTensor(X),
    #                        torch.FloatTensor(y))
    # train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    #
    # model.train()
    # for e in range(1, EPOCHS + 1):
    #     epoch_loss = 0
    #     epoch_acc = 0
    #     for X_batch, y_batch in train_loader:
    #         X_batch = X_batch.to(device)
    #         y_batch = y_batch.to(device)
    #
    #         optimizer.zero_grad()
    #
    #         y_pred = model(X_batch)
    #
    #         loss = criterion(y_pred, y_batch)
    #         acc = binary_acc(y_pred, y_batch.unsqueeze(1))
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss += loss.item()
    #         epoch_acc += acc.item()
    #
    #     print(
    #         f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(X):.5f} | Acc: {epoch_acc / len(X):.3f}')

    fpath = './models/neural-FNN-{}.pt'.format(EPOCHS)
    #
    # torch.save(model, fpath)
    model = torch.load(fpath)  # it is feedforward
    model = model.to(device)
    summary(model, input_size=(64, 4))
    print("Trained")

    plt.figure()
    plt.plot(all_losses)

    test_data = testData(torch.FloatTensor(X_test))
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(0 in y_pred_list)
    confusion_matrix(y_test, y_pred_list)
    print(classification_report(y_test, y_pred_list))

    print("Random")
    y_pred_list = [random.randint(0, 1) for _ in y_pred_list]
    confusion_matrix(y_test, y_pred_list)
    print(classification_report(y_test, y_pred_list))


if __name__ == '__main__':
    main()
