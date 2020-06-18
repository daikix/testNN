import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split as ttp
from tqdm import tqdm
import time
import collections


class MyDataSet(Dataset):
    def __init__(self, active=None):
        if active:
            self.active = active
            self.FolderPath = "/content/Activate"
            self.CSV = pd.read_csv(self.FolderPath + "/Activate.csv")
        else:
            self.active = active
            self.FolderPath = "/content/Dataset/"
            self.CSV = pd.read_csv(self.FolderPath + "MNIST.csv")

        self.transform, _ = MyTrans()

    def __len__(self):
        return len(self.CSV)

    def __getitem__(self, idx):
        FileName = self.CSV["id"][idx]
        label = np.array(self.CSV["label"][idx])
        if self.active:
            image = Image.open(self.FolderPath + "/data/" + str(FileName) + ".png")
        else:
            image = Image.open(self.FolderPath + "/data/" + str(label.item()) + "/" + FileName + ".png")
        if self.transform:
            image = self.transform(image)
            if self.active:
                image_act.append(np.array(image))

        return image, label


def MyTrans(image=None):
    size = (28, 28)
    tf = transforms.Compose([transforms.Grayscale(),
                             transforms.Resize(size),
                             transforms.ToTensor()])
    if image:
        image = tf(image)

    return tf, image


def flatten(array):
    for el in array:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


class MyNNet(nn.Module):
    def __init__(self):
        super(MyNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.drop2 = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    epoch = 300
    myDS = MyDataSet()
    train_data, test_data = ttp(myDS, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    MyNet = MyNNet()
    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MyNet.to(Device)
    Criterion = nn.CrossEntropyLoss()
    Optimizer = torch.optim.SGD(params=MyNet.parameters(), lr=0.001, momentum=0.9)
    result = {"train_loss": [], "test_acc": []}

    for e in range(epoch):
        loss = 0
        MyNet.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(Device)
            label = label.to(Device)
            Optimizer.zero_grad()
            output = MyNet(image)
            loss = Criterion(output, label)
            loss.backward()
            Optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch:{e + 1:03}  Train Process:{(i + 1) * 10:02} / {len(train_loader) * 10}")
                print(f"           Train Loss:{loss.item():05}")

        result["train_loss"].append(loss.item())

        MyNet.eval()
        correct = 0
        with torch.no_grad():
            for i, (image, label) in enumerate(tqdm(test_loader)):
                image = image.to(Device)
                label = label.to(Device)
                output = MyNet(image)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).sum().item()
                time.sleep(0.01)

            acc = float(correct / 200)
            result["test_acc"].append(acc)

    plt.plot(range(0, epoch), result["train_loss"])
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.close()

    plt.plot(range(0, epoch), result["test_acc"])
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    plt.close()

    print(f"Accuracy : {result['test_acc'][-1]:.5%}")

    torch.save(MyNet.to('cpu').state_dict(), "/content/model.pth")

    image_act = []
    myDS_Active = MyDataSet(True)
    batch = 10
    predicted_act = []
    label_act = []
    active_loader = DataLoader(myDS_Active, batch_size=batch, shuffle=True)
    MyNet.eval()
    for (image, label) in active_loader:
        image = image.to(Device)
        label = label.to(Device)
        label_act.append(label.tolist())
        output = MyNet(image)
        _, predicted = torch.max(output.data, 1)
        predicted_act.append(predicted.tolist())

    num = 1
    predicted_act = list(flatten(predicted_act))
    label_act = list(flatten(label_act))
    x = len(predicted_act)

    while True:
        if x - 5 < 0:
            break
        else:
            num += 1
            x = x - 5

    for idx, image in enumerate(image_act):
        plt.subplot(num, 5, idx + 1)
        plt.imshow(np.squeeze(np.transpose(np.array(image), (1, 2, 0))), cmap="gray")
        plt.subplots_adjust(hspace=0.9)
        plt.title(f"Result:{predicted_act[idx]}" + "\n" + f"Label:{label_act[idx]}", fontsize=12)
        # plt.tight_layout()
        plt.axis('off')

    count = 0
    if predicted_act == label_act: count += 1
    plt.show()
    print("正解数：" + str(count))