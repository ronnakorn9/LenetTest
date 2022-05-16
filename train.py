# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from src.lenet import Lenet

from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

### construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

### training hyperparameters
INIT_LR = 1e-3 # learning rate
BATCH_SIZE = 64
EPOCHS = 10

### define train/validation split portion
TRAIN_SPLIT = 0.75
VALI_SPLIT = 1 - TRAIN_SPLIT

### set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################
################  DATA SETUP  #################
###############################################
### get dataset
print("[INFO] Loading KMNIST dataset...")
train_data = KMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = KMNIST(root='data', train=False, download=True, transform=ToTensor())

### calculate the train/validation split
print("[INFO] Generating train/validation split...")
num_train = int(len(train_data) * TRAIN_SPLIT)
num_vali = int(len(train_data) * VALI_SPLIT)
(train_data, vali_data) = random_split(train_data, [num_train, num_vali], generator=torch.Generator().manual_seed(42))

### initialize the train, validation, and test data loaders
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
vali_data_loader = DataLoader(vali_data, batch_size=BATCH_SIZE)
test_data_loader= DataLoader(test_data, batch_size=BATCH_SIZE)

### calculate steps per epoch for training and validation set
train_steps = len(train_data_loader.dataset) // BATCH_SIZE
vali_steps = len(vali_data_loader.dataset) // BATCH_SIZE

##################################################
###############  MODEL BUILDING  #################
##################################################
### making Lenet model
model = Lenet(numChannels=1)
