import numpy as np
from torch import nn, optim
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from MODELS import MyModel, MyModel_small, MyModelwithAttention, MyModelwithAttention_small, MyModelwithAttention_factor4, MyModel_factor4
from netCDF4 import Dataset
from Dataset import TestingDataset, TestingDataset_factor4
import torchvision
from torchvision import transforms
from MODELS import MyModelwithAttention, MyModel
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

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

def RegulationCheck1(pred, HR):
    size1 = pred.shape[0]
    size2 = pred.shape[1]
    size3 = pred.shape[2]
    size4 = pred.shape[3]

    for i in range(size1):
        for j in range(size2):
            for m in range(size3):
                for n in range(size4):
                    if pred[i, j, m, n] > HR[i, j, m, n]:
                        pred[i, j, m, n] = HR[i, j, m, n]
    return pred

def ergodic_tensor(rito_tensor, yuzhi):
    shape = rito_tensor.shape
    num = 0
    for m in range(shape[0]):
        for k in range(shape[1]):
            for i in range(shape[2]):
                for j in range(shape[3]):
                    data = rito_tensor[m, k, i, j].item()
                    if data <= yuzhi:
                        num += 1
    return num

def PartAdd(data1, data2):
    size1 = data1.shape[2]
    size2 = data1.shape[3]
    data = torch.zeros(size=(data1.shape[0], 1, size1, size2))
    n = data1.shape[0]
    for k in range(n):
        for i in range(size1):
            for j in range(size2):
                if data1[k, 0, i, j] != 0:
                    data[k, 0, i, j] = data1[k, 0, i, j] + data2[k, 0, i, j]
    return data

def CountZero(data1 ,data2):     #hr pred
    n = data1.shape[1]
    size1 = data1.shape[2]
    size2 = data1.shape[3]
    numa = 0
    numb = 0
    numc = 0
    for N in range(data1.shape[0]):
        for k in range(n):
            for i in range(size1):
                for j in range(size2):
                    num1 = data1[N, k, i, j].item()
                    num2 = data2[N, k, i, j].item()
                    if num1 == 0.0000 and num2 == num1:
                        numa += 1
                    if num1 == 0.0000 and num1 < num2:
                        # print(data2[0, k, i, j].item())
                        numb += 1
                    if num2 == 0.0000 and num2 < num1:
                        numc += 1
                    # else:
                    #     print(data2[0, k, i, j].item())
    return numa, numb, numc

def CountAccu(pred, target, yuzhi):
    shape = pred.shape
    distribution = torch.zeros(size=(41, 41))
    for m in range(shape[0]):
        for k in range(shape[1]):
            for i in range(shape[2]):
                for j in range(shape[3]):
                    if target[m, k, i, j] == 0 and pred[m, k, i, j] == 0:
                        distribution[i, j] = 1
                    if torch.abs((target[m, k, i, j] - pred[m, k, i, j]) / target[m, k, i, j]) < yuzhi:
                        distribution[i, j] = 1
    return distribution

def CheckBase(target_top, pred_base):
    size1 = target_top.shape[0]
    size2 = target_top.shape[1]
    size3 = target_top.shape[2]
    size4 = target_top.shape[3]
    num2 = 0
    for i in range(size1):
        for j in range(size2):
            for m in range(1, size3 - 1):
                for n in range(1, size4 - 1):
                    if pred_base[i, j, m, n] > target_top[i, j, m, n]:
                        num2 += 1
    return num2

def CheckTop(target_base, pred_top):
    size1 = target_base.shape[0]
    size2 = target_base.shape[1]
    size3 = target_base.shape[2]
    size4 = target_base.shape[3]
    num1 = 0
    for i in range(size1):
        for j in range(size2):
            for m in range(1, size3 - 1):
                for n in range(1, size4 - 1):
                    if pred_top[i, j, m, n] < target_base[i, j, m, n]:
                        num1 += 1
    return num1

def testing(testing_type, model_structure, model_path1, model_path2="MyModelwithAttention_equal_small_pool53_test.pkl"):
    global test_dataset, model2, model1, predict_base, predict_difference
    nc = Dataset('202001nb.nc', mode='r')

    Base_Height = nc.variables['tplb'][:].astype(np.float32)
    Top_Height = nc.variables['tplt'][:].astype(np.float32)

    Base_Height = Tozero(Base_Height)
    Top_Height = Tozero(Top_Height)

    Base_Height_test = np.zeros(shape=(124, 41, 41))
    Top_Height_test = np.zeros(shape=(124, 41, 41))

    k = 0
    for i in range(0, 744):
        if i % 6 == 0:
            Base_Height_test[k] = Base_Height[i].astype(np.float32)
            Top_Height_test[k] = Top_Height[i].astype(np.float32)
            k += 1

    data_compose = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        if testing_type == 0:        #    0.5->0.25
            test_dataset = TestingDataset(Top_Height_test, Base_Height_test, data_compose)
            net = MyModelwithAttention.Model(model_structure[0], model_structure[1], model_structure[2],
                                             model_structure[3], model_structure[4], model_structure[5],
                                             model_structure[6], model_structure[7])
            model2 = net.to(device)
            checkpoint1 = torch.load(model_path1)
            model2.load_state_dict(checkpoint1['model_state_dict'])

        if testing_type == 1:       #    1->0.25
            test_dataset = TestingDataset_factor4(Top_Height_test, Base_Height_test, data_compose)
            net = MyModelwithAttention_factor4.Model(model_structure[0], model_structure[1], model_structure[2],
                                             model_structure[3], model_structure[4], model_structure[5],
                                             model_structure[6], model_structure[7])
            model2 = net.to(device)
            checkpoint1 = torch.load(model_path1)
            model2.load_state_dict(checkpoint1['model_state_dict'])
        if testing_type == 2:       #    1->0.25 stack
            test_dataset = TestingDataset_factor4(Top_Height_test, Base_Height_test, data_compose)
            net1 = MyModelwithAttention.Model(model_structure[0], model_structure[1], model_structure[2],
                                             model_structure[3], model_structure[4], model_structure[5],
                                             model_structure[6], model_structure[7])
            net2 = MyModelwithAttention_small.Model(model_structure[0], model_structure[1], model_structure[2],
                                             model_structure[3], model_structure[4], model_structure[5],
                                             model_structure[6], model_structure[7])
            model1 = net1.to(device)
            model2 = net2.to(device)
            checkpoint2 = torch.load(model_path2)
            checkpoint1 = torch.load(model_path1)
            model1.load_state_dict(checkpoint1['model_state_dict'])
            model2.load_state_dict(checkpoint2['model_state_dict'])
    except:
        print("error")

    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    bias_base = torch.zeros(41, 41)
    bias_base = bias_base.to(device)
    bias_top = torch.zeros(41, 41)
    bias_top = bias_top.to(device)

    top_loss = 0
    base_loss = 0

    L1 = nn.L1Loss()

    num_zero1 = 0
    num_zero2 = 0
    num_zero3 = 0
    num_zero4 = 0
    num_zero5 = 0
    num_zero6 = 0

    sum1 = 0
    sum2 = 0

    sum_01_1 = 0
    sum_01_2 = 0

    x = np.arange(110, 121, 10)
    y = np.arange(20, 9, -10)

    with torch.no_grad():
        j = 0
        for data in val_dataloader:
            lr_base = data['lr_base']
            hr_base = data['hr_base']
            lr_top = data['lr_top']
            hr_top = data['hr_top']
            lr_difference = data['lr_difference']
            hr_difference = data['hr_difference']

            lr_base = lr_base.to(device, torch.float32)
            hr_base = hr_base.to(device, torch.float32)
            lr_top = lr_top.to(device, torch.float32)
            hr_top = hr_top.to(device, torch.float32)
            lr_difference = lr_difference.to(device, torch.float32)
            hr_difference = hr_difference.to(device, torch.float32)

            predict_base, _ = model2(lr_base)
            judge_top1, _ = model2(lr_top)
            predict_base = RegulationCheck1(predict_base, judge_top1)

            predict_difference, _ = model2(lr_difference)

            if testing_type == 2:
                predict_base, _ = model1(predict_base)
                judge_top, _ = model1(judge_top1)
                predict_base = RegulationCheck1(predict_base, judge_top)

                predict_difference, _ = model1(predict_difference)

            predict_top = PartAdd(predict_base, predict_difference)

            predict_difference = predict_difference.to(device, torch.float32)
            predict_base = predict_base.to(device, torch.float32)
            predict_top = predict_top.to(device)

            bias1 = torch.abs(predict_base[0][0] - hr_base[0][0])
            bias2 = torch.abs(predict_top[0][0] - hr_top[0][0])
            bias_base += bias1
            bias_top += bias2

            top_loss += L1(predict_top, hr_top).item()
            base_loss += L1(predict_base, hr_base).item()

            a1, b1, c1 = CountZero(hr_top, predict_top)
            num_zero1 += a1  # hr and predict == 0
            num_zero2 += b1
            num_zero3 += c1

            a2, b2, c2 = CountZero(hr_base, predict_base)
            num_zero4 += a2
            num_zero5 += b2
            num_zero6 += c2

            rito1 = torch.abs((predict_base - hr_base) / hr_base)
            sum1 += ergodic_tensor(rito1, 0.2)
            sum_01_1 += ergodic_tensor(rito1, 0.1)
            AccuDis1 = CountAccu(predict_base, hr_base, 0.2)
            AccuDis3 = CountAccu(predict_base, hr_base, 0.1)
            # AccuDis1 = AccuDistribution(rito1, 0.2)
            #
            rito = torch.abs((predict_top - hr_top) / hr_top)
            sum2 += ergodic_tensor(rito, 0.2)
            sum_01_2 += ergodic_tensor(rito, 0.1)
            AccuDis2 = CountAccu(predict_top, hr_top, 0.2)
            AccuDis4 = CountAccu(predict_top, hr_top, 0.1)

            if j == 312:
                plt.figure()
                plt.subplot(2, 5, 1)
                plt.imshow(hr_base[0][0].cpu(), cmap=plt.get_cmap('jet'))
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("base GroundTruth")
                plt.subplot(2, 5, 2)
                plt.imshow(predict_base[0][0].cpu(), cmap=plt.get_cmap('jet'))
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("base prediction")
                plt.subplot(2, 5, 3)
                plt.imshow(bias1.cpu(), cmap=plt.get_cmap('jet'))
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("base bias")
                plt.subplot(2, 5, 4)
                plt.imshow(AccuDis1.cpu())
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("base bias 0.2")
                plt.subplot(2, 5, 5)
                plt.imshow(AccuDis3.cpu())
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("base bias 0.1")

                plt.subplot(2, 5, 6)
                plt.imshow(hr_top[0][0].cpu(), cmap=plt.get_cmap('jet'))
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("top GroundTruth")
                plt.subplot(2, 5, 7)
                plt.imshow(predict_top[0][0].cpu(), cmap=plt.get_cmap('jet'))
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("top prediction")
                plt.subplot(2, 5, 8)
                plt.imshow(bias2.cpu(), cmap=plt.get_cmap('jet'))
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title(j)
                plt.subplot(2, 5, 9)
                plt.imshow(AccuDis2.cpu())
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("top bias 0.2")
                plt.subplot(2, 5, 10)
                plt.imshow(AccuDis4.cpu())
                plt.xticks(np.arange(0, 41, 40), x)
                plt.yticks(np.arange(0, 41, 40), y)
                plt.title("top bias 0.1")
                plt.tight_layout()
                plt.show()

            if L1(predict_base, hr_base).item() > 0:
                print("No.", j, " loss:", L1(predict_base, hr_base).item(), "Accu1:",
                      (a2 + ergodic_tensor(rito1, 0.2)) / (41 * 41), "Accu2:",
                      (a2 + ergodic_tensor(rito1, 0.1)) / (41 * 41))

            j += 1

    print("Top:")
    print(sum2)
    print((sum2 + num_zero1) / (124 * 41 * 41))  # 416888
    print(sum_01_2)
    print((sum_01_2 + num_zero1) / (124 * 41 * 41))
    print("hr和predict都是0:", num_zero1)
    print("hr是0，predict不是:", num_zero2)
    print("hr不是0，predict是:", num_zero3)
    print("loss:", top_loss / 124)
    print("Base:")
    print(sum1)
    print((sum1 + num_zero4) / (124 * 41 * 41))
    print(sum_01_1)
    print((sum_01_1 + num_zero4) / (124 * 41 * 41))
    print("hr和predict都是0:", num_zero4)
    print("hr是0，predict不是:", num_zero5)
    print("hr不是0，predict是:", num_zero6)
    print("Base_loss:", base_loss / 124)

if __name__ == '__main__':
    # testing(0, [128, 5, 3, 3, 3, 3, 5, 3], "MyModelwithAttention_equal_type0_128_5_pool33.pkl")
    testing(1, [128, 5, 3, 3, 3, 3, 5, 3], "MyModelwithAttention_equal_type1_128_5_pool53.pkl")
    # testing(2, [128, 5, 3, 3, 3, 3, 5, 3], "MyModelwithAttention_equal_type0_128_5_pool33.pkl", "MyModelwithAttention_equal_type2_128_5_pool53.pkl")







