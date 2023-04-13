import numpy as np
from torch import nn, optim
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from MODELS import MyModelwithAttention, MyModelwithAttention_factor4, MyModel, MyModel_small, MyModel_factor4, MyModelwithAttention_small
from netCDF4 import Dataset
from Dataset import TrainingDataset, TrainingDataset_factor4, TrainingDataset_small
import torch.nn.functional as F
import time
import os
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

def Tozero(data):
    n = data.shape[0]
    size1 = data.shape[1]
    size2 = data.shape[2]
    for k in range(n):
        for i in range(size1):
            for j in range(size2):
                if data[k, i, j] < 0:
                    data[k, i, j] = 0
    return data


def training(model_type, model_structure, batch_size, epoch, down_epoch):
    global dataset, val_dataset, net
    nc = Dataset('202001nb.nc', mode='r')
    Base_Height = nc.variables['tplb'][:].astype(np.float32)  # [:, 16:41, 0:21]
    Top_Height = nc.variables['tplt'][:].astype(np.float32)

    Base_Height = Tozero(Base_Height)
    Top_Height = Tozero(Top_Height)

    Base_Height_Train1 = np.zeros(shape=(620, 41, 41))  # 620, 41, 41      620, 25, 21
    Top_Height_Train1 = np.zeros(shape=(620, 41, 41))
    Base_Height_Val = np.zeros(shape=(124, 41, 41))
    Top_Height_Val = np.zeros(shape=(124, 41, 41))

    j = 0
    k = 0
    for i in range(0, 744):
        if i % 6 != 0:
            Base_Height_Train1[j] = Base_Height[i].astype(np.float32)
            Top_Height_Train1[j] = Top_Height[i].astype(np.float32)
            j += 1
        else:
            Base_Height_Val[k] = Base_Height[i].astype(np.float32)
            Top_Height_Val[k] = Top_Height[i].astype(np.float32)
            k += 1

    data_compose = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])

    try:
        if model_type == 0:      #0.5->0.25
            dataset = TrainingDataset(Top_Height_Train1, Base_Height_Train1, data_compose)
            val_dataset = TrainingDataset(Top_Height_Val, Base_Height_Val, data_compose)
            net = MyModelwithAttention.Model(model_structure[0], model_structure[1], model_structure[2],
                                             model_structure[3], model_structure[4], model_structure[5],
                                             model_structure[6], model_structure[7])

        if model_type == 1:      #1->0.25
            dataset = TrainingDataset_factor4(Top_Height_Train1, Base_Height_Train1, data_compose)
            val_dataset = TrainingDataset_factor4(Top_Height_Val, Base_Height_Val, data_compose)
            net = MyModelwithAttention_factor4.Model(model_structure[0], model_structure[1], model_structure[2],
                                                     model_structure[3], model_structure[4], model_structure[5],
                                                     model_structure[6], model_structure[7])
        if model_type == 2:      #1->0.5
            dataset = TrainingDataset_small(Top_Height_Train1, Base_Height_Train1, data_compose)
            val_dataset = TrainingDataset_small(Top_Height_Val, Base_Height_Val, data_compose)
            net = MyModelwithAttention_small.Model(model_structure[0], model_structure[1], model_structure[2],
                                                   model_structure[3], model_structure[4], model_structure[5],
                                                   model_structure[6], model_structure[7])
    except:
        print("error")

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)

    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    total_epoch = epoch
    criterion = nn.SmoothL1Loss()

    path = "MyModelwithAttention_equal_type{}_{}_{}_pool{}{}.pkl".format(model_type, model_structure[0], model_structure[1], model_structure[6], model_structure[7])

    if os.path.exists(path):
        print("continue training")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        Best_loss = checkpoint['min_loss']
    else:
        print("Init training")
        start_epoch = 0
        Best_loss = 100000
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, val=0.0)



    time_start = time.time()
    model.train()

    # for param_group in optimizer.param_groups:
    #     param_group["lr"] *= 0.1

    for epoch in range(start_epoch, total_epoch):
        if epoch % down_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1
        i = 0
        pred_loss = 0
        j = 0
        for data in trainloader:
            lr = data['lr_base']
            hr = data['hr_base']

            lr = Variable(lr.to(device, torch.float32), requires_grad=True)
            hr = Variable(hr.to(device, torch.float32), requires_grad=True)

            predict, edge = model(lr)
            loss = criterion(predict, hr)
            pred_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            j = j + 1
        avg_val_loss = 0
        m = 0
        for data in val_dataloader:
            lr = data['lr_base']
            hr = data['hr_base']

            lr = Variable(lr.to(device, torch.float32), requires_grad=False)
            hr = Variable(hr.to(device, torch.float32), requires_grad=False)

            with torch.no_grad():
                predict_val, edge = model(lr)
                avg_val_loss += criterion(predict_val, hr).item()
            m += 1
        val_loss = avg_val_loss / 124
        if val_loss < Best_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                ""
                "epoch": epoch,
                "min_loss": val_loss
            }
            Best_loss = val_loss
            torch.save(checkpoint, path)
        print("epoch:", epoch, " turn:", i, " loss:", pred_loss / (620 / batch_size), "val_loss:", val_loss, "Best_val_loss:",
              Best_loss, "learning rate:",
              optimizer.state_dict()['param_groups'][0]['lr'])
    model.eval()
    time_end = time.time()
    print("cost ", (time_end - time_start), " s")

if __name__ == '__main__':
    # training(0, [128, 5, 3, 3, 3, 3, 3, 3], 124, 7000, 5000)
    # training(1, [128, 5, 3, 3, 3, 3, 5, 3], 124, 7000, 5000)
    training(2, [128, 5, 3, 3, 3, 3, 5, 3], 124, 7000, 5000)
