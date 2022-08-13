#!/usr/bin/python3
#coding=utf-8

import os
# from typing import final
import cv2
import torch
import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset
import os


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask, corner_mask=None):
        image = (image - self.mean)/self.std
        #print(mask.shape)
        mask /= 255
        if corner_mask is not None:
            #print(corner_mask.shape)
            for i in range(len(corner_mask)):
                corner_mask[i] = corner_mask[i] / 255
            return image, mask, corner_mask
        else:
            return image, mask

class RandomCrop(object):
    def __call__(self, image, mask, corner_mask=None):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if corner_mask is not None:
            corner_mask_ = []
            for e in corner_mask:
                corner_mask_.append(e[p0:p1, p2:p3])
            return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], corner_mask_

class RandomFlip(object):
    def __call__(self, image, mask, corner_mask=None):
        if corner_mask is not None:
            if np.random.randint(2)==0:
                corner_mask_ = []
                for e in corner_mask:
                    corner_mask_.append(e[:, ::-1])
                return image[:,::-1,:], mask[:, ::-1], corner_mask_
            else:
                return image, mask, corner_mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, corner_mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if corner_mask is not None:
            for i in range(len(corner_mask)):
                corner_mask[i] = cv2.resize(corner_mask[i], dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask, corner_mask
        else:
            return image, mask

class ToTensor(object):
    def __call__(self, image, mask, corner_mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        if corner_mask is not None:
            for i in range(len(corner_mask)):
                corner_mask[i] = torch.from_numpy(corner_mask[i])
            return image, mask, corner_mask
        else:
            return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(544, 544)
        self.totensor   = ToTensor()
        #print(cfg.mode)
        with open(os.path.join(cfg.datapath, cfg.mode+'.txt'), 'r') as lines:
            self.samples = []
            for line in lines:
               # print(line.split()[0])
                self.samples.append(line.strip())

    def __getitem__(self, idx):

        if self.cfg.mode=='train_cam' or self.cfg.mode == 'train_sal':
            im_name = self.samples[idx].split()[0]
            gt_name = self.samples[idx].split()[1]
            corner_name = self.samples[idx].split()[2]

            image = cv2.imread(os.path.join(self.cfg.datapath, im_name))[:, :, ::-1].astype(np.float32)
            g_image = cv2.imread(os.path.join(self.cfg.datapath, im_name), 0)
            sal_label = cv2.imread(os.path.join(self.cfg.datapath, gt_name), 0).astype(np.float32)
            sal_corner_label = cv2.imread(os.path.join(self.cfg.datapath, corner_name), 0).astype(np.float32)
            sal_corner_label = (sal_label > 127).astype(np.uint8) * sal_corner_label
            mask = ((sal_label > 127).astype(np.uint8) * 255).astype(np.float32)
            bg_mask = (sal_label <= 127).astype(np.uint8)
            corner_mask = ((sal_corner_label > 77).astype(np.uint8) * 255).astype(np.float32)
            nums = np.count_nonzero(sal_corner_label > 77)
            nums_hard = np.count_nonzero(sal_corner_label > 229)
            corner_mask_hard = sal_corner_label#(sal_corner_label > 229).astype(np.uint8) * 255
            nums_mid = np.count_nonzero(sal_corner_label > 127)
            corner_mask_mid = (sal_corner_label > 127).astype(np.uint8) * 255
            corner_mask_hardmid = (sal_corner_label > 179).astype(np.uint8) * 255
            flag_hard = (bg_mask+(sal_corner_label > 229).astype(np.uint8)) * 255
            flag_hardmid = (bg_mask + (sal_corner_label > 179).astype(np.uint8)) * 255
            flag_mid = (bg_mask + (sal_corner_label > 127).astype(np.uint8)) * 255
            flag = (bg_mask + (sal_corner_label > 77).astype(np.uint8)) * 255


            '''
            if nums > 0:
                flag = np.ones_like(corner_mask)
            else:
                flag = np.zeros_like(corner_mask)
            if nums_hard > 0:
                flag_hard = np.ones_like(corner_mask)
            else:
                flag_hard = np.zeros_like(corner_mask)
            if nums_mid > 0:
                flag_mid = np.ones_like(corner_mask)
            else:
                flag_mid = np.zeros_like(corner_mask)
            '''
            image, mask, [corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image] = self.normalize(
                image, mask, [corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image])
            image, mask, [corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image] = self.randomcrop(
                image, mask, [corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image])
            image, mask, [corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image] = self.randomflip(
                image, mask, [corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image])

            return image, mask, corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image
        

        else:
            im_name = self.samples[idx].split()[0]
            #print(im_name)
            image = cv2.imread(os.path.join(self.cfg.datapath,'Imgs',im_name + '.jpg'))[:, :, ::-1].astype(np.float32)
            #mask = cv2.imread(os.path.join(self.cfg.datapath.replace('Image', 'GT'), im_name.replace('.jpg', '.png')), 0).astype(np.float32)
            mask = cv2.imread(os.path.join(self.cfg.datapath,'GT', im_name + '.png'), 0).astype(np.float32)
            shape = mask.shape
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, im_name.split('/')[-1]

    def collate(self, batch):
        
        size = [384,416,480,512,544][np.random.randint(0, 5)]
        ### Origin
        image, mask, corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            corner_mask[i] = cv2.resize(corner_mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            flag[i] = cv2.resize(flag[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            corner_mask_hard[i] = cv2.resize(corner_mask_hard[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            flag_hard[i] = cv2.resize(flag_hard[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            corner_mask_mid[i] = cv2.resize(corner_mask_mid[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            flag_mid[i] = cv2.resize(flag_mid[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            corner_mask_hardmid[i] = cv2.resize(corner_mask_hardmid[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            flag_hardmid[i] = cv2.resize(flag_hardmid[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            g_image[i] = cv2.resize(g_image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        corner_mask = torch.from_numpy(np.stack(corner_mask, axis=0)).unsqueeze(1)
        flag = torch.from_numpy(np.stack(flag, axis=0)).unsqueeze(1)
        corner_mask_hard = torch.from_numpy(np.stack(corner_mask_hard, axis=0)).unsqueeze(1)
        flag_hard = torch.from_numpy(np.stack(flag_hard, axis=0)).unsqueeze(1)
        corner_mask_mid = torch.from_numpy(np.stack(corner_mask_mid, axis=0)).unsqueeze(1)
        flag_mid = torch.from_numpy(np.stack(flag_mid, axis=0)).unsqueeze(1)
        corner_mask_hardmid = torch.from_numpy(np.stack(corner_mask_hardmid, axis=0)).unsqueeze(1)
        flag_hardmid = torch.from_numpy(np.stack(flag_hardmid, axis=0)).unsqueeze(1)
        g_image = torch.from_numpy(np.stack(g_image, axis=0)).unsqueeze(1)
        return image, mask, corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image

        
    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='E:\\SalientObject\\testdata\\DUTS-TR\\Imgs')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
