from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch
import random
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

train_path = './train'
# test_path = './train'

seed = 3074
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)
def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

train_transform = transforms.Compose([
 #   transforms.RandomRotation(15),
    transforms.Resize([384, 384]),
   # transforms.RandomVerticalFlip(),
    FixedRotation([0, 90, 180, -90]),
    #transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
])
val_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536))
])

train_transform2 = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.Resize([256,256 ]),
    transforms.ToTensor(),
    transforms.Normalize((0.3705, 0.3828, 0.3545), (0.1685, 0.1590, 0.1536)),
])

train_dataset=ImageFolder(train_path,transform=train_transform2)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True, num_workers=0)

# test_dataset=ImageFolder(test_path,transform=val_transform)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64,shuffle=None, num_workers=0)
model_name = "mobilevit_s"
epoch_num=10
net=timm.create_model(model_name,pretrained=True,num_classes=5).cuda()
# weight = torch.load('./efficientnet_b0lab_8.pth')
# net.load_state_dict(weight['model_state_dict'])
criterion=nn.CrossEntropyLoss()

# weight_decay = 1e-4  # L2 正则化系数
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
#lr_scheduler = CosineLRScheduler(optimizer, t_initial=0.02, lr_min=0.000004)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
PATH = './'+model_name  #这里也需要注意，你需要先新建一个dir文件夹，一会在这个文件下存放权重
pre_acc = 0
def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    val_loss /= total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc
best_val_acc=0
if __name__ == '__main__':
    correct = 0
    total = 0
    for epoch in range(epoch_num):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, epoch_num))
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            train_acc = 100.0 * correct / total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} train acc: {:.6f}  lr : {:.6f}'.format(epoch+1, i * len(inputs),
                                                                                                              len(trainloader.dataset),100. * i / len(trainloader),loss.item(),train_acc,get_cur_lr(optimizer)))
        if epoch % 1 ==0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        },'./nfnet/'+PATH + "_"+str(epoch)+".pth")
        # print('===================test============================')
        # val_loss, val_acc = validate(net, testloader, criterion)
        # print(f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_acc:.6f}')
        
        # Save the model if it has the best accuracy so far
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        # torch.save({'epoch': epoch,
        #             'model_state_dict': net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             }, './weights2/'+PATH + "_best.pth")
        
        # Save the model checkpoint
        
        
        lr_scheduler.step()

