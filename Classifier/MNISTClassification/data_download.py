import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from pathlib import Path
import shutil

print(f"Torch Version: {torch.__version__} \nTorchvision Version: {torchvision.__version__}")

data_path = Path("../data")

if data_path.exists():
    print(f'[INFO] Clearing existing data path.')
    shutil.rmtree(data_path)
else:
    data_path.mkdir(parents=True, exist_ok=False)


print(f"Donloading data to {data_path}")
#Setting up the train data:
train_data = datasets.MNIST(root= data_path,
                            train = True,
                            download= True,
                            transform=ToTensor(),
                            target_transform=None)

#Setting up train data:
test_data = datasets.MNIST(root = data_path,
                           train = False,
                            download= True,
                            transform=ToTensor(),
                            target_transform=None)