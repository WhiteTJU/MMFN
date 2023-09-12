import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import json
import os
from PIL import Image
import numpy as np

class GetDataTrain(Dataset):
    def __init__(self, dataType='train', imageMode='RGB',views=20):
        self.dt = dataType
        self.imd = imageMode
        self.views = views 
        # path_dm = "/home/zdd/data/pc/12_ModelNet40_depth/"
        # path = "/home/zdd/data/modelnet40v2png_ori4_view3/"
        path_dm = "/home/zdd/data/modelnet40v2png_ori4/"
        
        clses = sorted(os.listdir(path_dm))
        # clses = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        # 1 2 8 12 14 22 23 30 33 35

        self.fls_dm = []
        self.las_dm = []
        self.names_dm = []
        for c,cls in enumerate(clses):
            
            cls_path_dm = os.path.join(path_dm, cls, self.dt)
            objes = sorted(os.listdir(cls_path_dm))

            for i in range(len(objes)//20):
                views_path_dm = np.array([os.path.join(cls_path_dm, v) for v in objes[i*20 :i*20+20]])
                self.fls_dm.append(views_path_dm)
                self.las_dm.append(c)
                self.names_dm.append(views_path_dm[0])
        self.fls_dm = np.array(self.fls_dm)
        self.las_dm = np.array(self.las_dm)

        #
        path_mv = '/home/zdd/data/Modelnet40_view20_dm'
        # path = '/home/zdd/data/Modelnet10_view20'
        clses = sorted(os.listdir(path_mv))
        # clses = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        self.fls_mv = []
        self.las_mv = []
        self.names_mv = []
        for c,cls in enumerate(clses):
            
            cls_path_mv = os.path.join(path_mv, cls, self.dt)
            objes = sorted(os.listdir(cls_path_mv)) 
            #print('%d'%c,len(objes))
            for obj in objes:
                obj_path_mv = os.path.join(cls_path_mv, obj)
                views_path_mv = np.array([os.path.join(obj_path_mv, v) for v in sorted(os.listdir(obj_path_mv))])
                self.fls_mv.append(views_path_mv)
                self.las_mv.append(c)
                self.names_mv.append(obj_path_mv)
        self.fls_mv = np.array(self.fls_mv)
        self.las_mv = np.array(self.las_mv)




    
    def trans(self, path):
        img = Image.open(path).convert('RGB')
        tf = transforms.Compose([
                transforms.RandomRotation(degrees=8), # fill=234),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        img = tf(img)
        return img
  

    def __getitem__(self, index):
        fl_dm = self.fls_dm[index][:self.views]
        target_dm = torch.LongTensor([self.las_dm[index]])
        imgs_dm = []
        for p in fl_dm:
            imgs_dm.append(self.trans(p))
        data_dm = torch.stack(imgs_dm) 
        #
        fl_mv = self.fls_mv[index][:self.views]
        target_mv = torch.LongTensor([self.las_mv[index]])
        imgs_mv = []
        for p in fl_mv:
            imgs_mv.append(self.trans(p))
        data_mv = torch.stack(imgs_mv) 

        return {'data_mv':data_dm, 'data_dm':data_mv, 'target_mv':target_dm, 'target_dm':target_mv, 'name_mv': self.names_dm[index], 'name_dm': self.names_mv[index]}
                # data, big class , fine class,  domain

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.las_dm)


class GetDataTsne(Dataset):
    def __init__(self, dataType='train', imageMode='RGB', views=20, tsne=False):
        self.tsne = tsne
        self.dt = dataType
        self.imd = imageMode
        self.views = views
        # path_dm = "/home/zdd/data/pc/12_ModelNet40_depth/"
        # path = "/home/zdd/data/modelnet40v2png_ori4_view3/"
        path_dm = "/home/zdd/data/tsne_m40_v20_VD/V/"

        clses = sorted(os.listdir(path_dm))
        self.fls_dm = []
        self.las_dm = []
        self.names_dm = []
        for c, cls in enumerate(clses):

            cls_path_dm = os.path.join(path_dm, cls, self.dt)
            objes = sorted(os.listdir(cls_path_dm))

            for i in range(len(objes) // 20):
                views_path_dm = np.array([os.path.join(cls_path_dm, v) for v in objes[i * 20:i * 20 + 20]])
                self.fls_dm.append(views_path_dm)
                self.las_dm.append(c)
                self.names_dm.append(views_path_dm[0])
        self.fls_dm = np.array(self.fls_dm)
        self.las_dm = np.array(self.las_dm)

        #
        path_mv = '/home/zdd/data/tsne_m40_v20_VD/D'
        # path = '/home/zdd/data/Modelnet10_view20'
        clses = sorted(os.listdir(path_mv))

        self.fls_mv = []
        self.las_mv = []
        self.names_mv = []
        for c, cls in enumerate(clses):

            cls_path_mv = os.path.join(path_mv, cls, self.dt)
            objes = sorted(os.listdir(cls_path_mv))
            # print('%d'%c,len(objes))
            for obj in objes:
                obj_path_mv = os.path.join(cls_path_mv, obj)
                views_path_mv = np.array([os.path.join(obj_path_mv, v) for v in sorted(os.listdir(obj_path_mv))])
                self.fls_mv.append(views_path_mv)
                self.las_mv.append(c)
                self.names_mv.append(obj_path_mv)
        self.fls_mv = np.array(self.fls_mv)
        self.las_mv = np.array(self.las_mv)

    def trans(self, path):
        img = Image.open(path).convert('RGB')
        tf = transforms.Compose([
            transforms.RandomRotation(degrees=8),  # fill=234),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        return img

    def __getitem__(self, index):
        fl_dm = self.fls_dm[index][:self.views]
        target_dm = torch.LongTensor([self.las_dm[index]])
        imgs_dm = []
        for p in fl_dm:
            imgs_dm.append(self.trans(p))
        data_dm = torch.stack(imgs_dm)
        #
        fl_mv = self.fls_mv[index][:self.views]
        target_mv = torch.LongTensor([self.las_mv[index]])
        imgs_mv = []
        for p in fl_mv:
            imgs_mv.append(self.trans(p))
        data_mv = torch.stack(imgs_mv)

        return {'data_mv': data_dm, 'data_dm': data_mv, 'target_mv': target_dm, 'target_dm': target_mv,
                'name_mv': self.names_dm[index], 'name_dm': self.names_mv[index]}
        # data, big class , fine class,  domain

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.las_dm)


if __name__ == '__main__':
    a = GetDataTrain(dataType='train', imageMode='RGB')
    for i,d in enumerate(a):
        print('i:%d / %d'%(i,len(a)),d['data'].shape, d['target'], d['name'])



# 测试方法
# 1先输出标签和name查看是否对应
# 2 输出某个视图，查看是否对应
