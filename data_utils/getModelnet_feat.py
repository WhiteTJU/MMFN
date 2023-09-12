import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import json
import os
from PIL import Image
import numpy as np

def getdata(dataType):
    # path = '/home/dh/zdd/DAN/modelnet40_features_shuffle_vgg11_12.pth'
    train_path = 'C:/Users/16053/Documents/Tencent Files/1605322728/FileRecv/train.pth'
    test_path = 'C:/Users/16053/Documents/Tencent Files/1605322728/FileRecv/test.pth'
    if dataType == 'train':
        path = train_path
    else:
        path = test_path
    data_all = torch.load(path)
    print('data:', data_all.keys()) #  dict_keys(['class', 'train_data', 'tr    ain_label', 'val_data', 'val_label'])
    return data_all


class GetDataset40(Dataset):
    def __init__(self, dataType):
        data = getdata(dataType)
        if dataType == 'train':
            self.value = data['train_data']
            self.label = data['train_label']
        elif dataType == 'test':
            self.value = data['val_data']
            self.label = data['val_label']
        else:
            print('Format is wrong!')

    def __getitem__(self,index):
        return {'data':self.value[index], 'target':self.label[index]}

    def __len__(self):
        return len(self.label)


class GetDataset10(Dataset):
    def __init__(self, dataType):
        self.dt = dataType
        # path = '/home/dh/zdd/DAN/baseline/data/modelnet10/%s.pth' % self.dt
        path = 'C:/Users/16053/Documents/Tencent Files/1605322728/FileRecv/%s.pth' % self.dt
        # path = '/home/dh/zdd/DAN/modelnet40_vgg11_4096/%s.pth'% self.dt
        data = torch.load(path)
        self.value = data['data']
        self.label = data['target']
    def __getitem__(self, index):
        return {'data': self.value[index], 'target': self.label[index]}

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    a = GetDataTrain(path='', dataType='train', imageMode='RGB', version='16')
    for i, d in enumerate(a):
        print('i:%d / %d'%(i,len(a)),d['data'].shape, d['target'], d['name'])



# 测试方法
# 1先输出标签和name查看是否对应
# 2 输出某个视图，查看是否对应
