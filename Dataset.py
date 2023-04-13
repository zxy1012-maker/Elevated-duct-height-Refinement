import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image

class TrainingDataset(Dataset):
    def __init__(self, feature1, feature2, data_compose):
        super(TrainingDataset, self).__init__()
        self.lr_top = torch.tensor(feature1[::, ::2, ::2])
        self.hr_top = torch.tensor(feature1)
        self.lr_base = torch.tensor(feature2[::, ::2, ::2])
        self.hr_base = torch.tensor(feature2)
        self.Compose_data = data_compose

    def __getitem__(self, item):
        LR_top = Image.fromarray(self.lr_top[item].numpy())
        LR_base = Image.fromarray(self.lr_base[item].numpy())
        HR_top = Image.fromarray(self.hr_top[item].numpy())
        HR_base = Image.fromarray(self.hr_base[item].numpy())

        top_data = self.Compose_data(LR_top)
        top_target = self.Compose_data(HR_top)

        base_data = self.Compose_data(LR_base)
        base_target = self.Compose_data(HR_base)
        base_truth = torch.tensor(self.hr_base)

        HR_LR = {
            "lr_base": base_data, "hr_base": base_target, "lr_top": top_data, "hr_top": top_target, "truth": base_truth[item]
        }
        return HR_LR

    def __len__(self):
        return len(self.lr_top)

class TestingDataset(Dataset):
    def __init__(self, feature1, feature2, data_compose):
        super(TestingDataset, self).__init__()
        self.lr_top = torch.tensor(feature1[::, ::2, ::2])
        self.hr_top = torch.tensor(feature1)
        self.lr_base = torch.tensor(feature2[::, ::2, ::2])
        self.hr_base = torch.tensor(feature2)
        self.lr_difference = self.lr_top - self.lr_base
        self.hr_difference = self.hr_top - self.hr_base
        self.Compose_data = data_compose

    def __getitem__(self, item):
        LR_top = Image.fromarray(self.lr_top[item].numpy())
        LR_base = Image.fromarray(self.lr_base[item].numpy())
        LR_difference = Image.fromarray(self.lr_difference[item].numpy())
        HR_top = Image.fromarray(self.hr_top[item].numpy())
        HR_base = Image.fromarray(self.hr_base[item].numpy())
        HR_difference = Image.fromarray(self.hr_difference[item].numpy())

        top_data = self.Compose_data(LR_top)
        top_target = self.Compose_data(HR_top)

        base_data = self.Compose_data(LR_base)
        base_target = self.Compose_data(HR_base)

        difference_data = self.Compose_data(LR_difference)
        difference_target = self.Compose_data(HR_difference)

        HR_LR = {
            "lr_base": base_data, "hr_base": base_target, "lr_top": top_data, "hr_top": top_target, "lr_difference": difference_data, "hr_difference": difference_target
        }

        return HR_LR

    def __len__(self):
        return len(self.hr_top)


class TrainingDataset_small(Dataset):
    def __init__(self, feature1, feature2, data_compose):
        super(TrainingDataset_small, self).__init__()
        self.lr_top = torch.tensor(feature1[::, ::2, ::2][::, ::2, ::2])
        self.hr_top = torch.tensor(feature1[::, ::2, ::2])
        self.lr_base = torch.tensor(feature2[::, ::2, ::2][::, ::2, ::2])
        self.hr_base = torch.tensor(feature2[::, ::2, ::2])
        self.Compose_data = data_compose

    def __getitem__(self, item):
        # LR_top = self.lr_top[item]
        # LR_base = self.lr_base[item]
        # HR_top = self.hr_top[item]
        # HR_base = self.hr_base[item]
        LR_top = Image.fromarray(self.lr_top[item].numpy())
        LR_base = Image.fromarray(self.lr_base[item].numpy())
        HR_top = Image.fromarray(self.hr_top[item].numpy())
        HR_base = Image.fromarray(self.hr_base[item].numpy())

        top_data = self.Compose_data(LR_top)
        top_target = self.Compose_data(HR_top)

        base_data = self.Compose_data(LR_base)
        base_target = self.Compose_data(HR_base)
        base_truth = torch.tensor(self.hr_base)

        HR_LR = {
            "lr_base": base_data, "hr_base": base_target, "lr_top": top_data, "hr_top": top_target, "truth": base_truth[item]
        }
        return HR_LR

    def __len__(self):
        return len(self.lr_top)

class TrainingDataset_factor4(Dataset):
    def __init__(self, feature1, feature2, data_compose):
        super(TrainingDataset_factor4, self).__init__()
        self.lr_top = torch.tensor(feature1[::, ::2, ::2][::, ::2, ::2])
        self.hr_top = torch.tensor(feature1)
        self.lr_base = torch.tensor(feature2[::, ::2, ::2][::, ::2, ::2])
        self.hr_base = torch.tensor(feature2)
        self.Compose_data = data_compose

    def __getitem__(self, item):
        LR_top = Image.fromarray(self.lr_top[item].numpy())
        LR_base = Image.fromarray(self.lr_base[item].numpy())
        HR_top = Image.fromarray(self.hr_top[item].numpy())
        HR_base = Image.fromarray(self.hr_base[item].numpy())

        top_data = self.Compose_data(LR_top)
        top_target = self.Compose_data(HR_top)

        base_data = self.Compose_data(LR_base)
        base_target = self.Compose_data(HR_base)
        base_truth = torch.tensor(self.hr_base)

        HR_LR = {
            "lr_base": base_data, "hr_base": base_target, "lr_top": top_data, "hr_top": top_target, "truth": base_truth[item]
        }
        return HR_LR

    def __len__(self):
        return len(self.lr_top)

class TestingDataset_factor4(Dataset):
    def __init__(self, feature1, feature2, data_compose):
        super(TestingDataset_factor4, self).__init__()
        self.lr_top = torch.tensor(feature1[::, ::2, ::2][::, ::2, ::2])
        self.hr_top = torch.tensor(feature1)
        self.lr_base = torch.tensor(feature2[::, ::2, ::2][::, ::2, ::2])
        self.hr_base = torch.tensor(feature2)
        self.lr_difference = self.lr_top - self.lr_base
        self.hr_difference = self.hr_top - self.hr_base
        self.Compose_data = data_compose

    def __getitem__(self, item):
        LR_top = Image.fromarray(self.lr_top[item].numpy())
        LR_base = Image.fromarray(self.lr_base[item].numpy())
        LR_difference = Image.fromarray(self.lr_difference[item].numpy())
        HR_top = Image.fromarray(self.hr_top[item].numpy())
        HR_base = Image.fromarray(self.hr_base[item].numpy())
        HR_difference = Image.fromarray(self.hr_difference[item].numpy())

        top_data = self.Compose_data(LR_top)
        top_target = self.Compose_data(HR_top)

        base_data = self.Compose_data(LR_base)
        base_target = self.Compose_data(HR_base)

        difference_data = self.Compose_data(LR_difference)
        difference_target = self.Compose_data(HR_difference)

        HR_LR = {
            "lr_base": base_data, "hr_base": base_target, "lr_top": top_data, "hr_top": top_target, "lr_difference": difference_data, "hr_difference": difference_target
        }

        return HR_LR

    def __len__(self):
        return len(self.hr_top)

