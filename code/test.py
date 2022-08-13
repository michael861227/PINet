#!/usr/bin/python3
#coding=utf-8

import os
import sys

from numpy.core.fromnumeric import size
from torch.nn.modules.activation import Softmax
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset

import argparse
from network import PINet




class Test(object):
    def __init__(self, Dataset, Network, datapath, modelpath):
        ## dataset
        self.cfg    = Dataset.Config(datapath=datapath, snapshot=modelpath, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False)
        
        ## network
        self.net =  Network()
        
        self.net.load_state_dict(torch.load(self.cfg.snapshot, map_location='cpu'))
        pytorch_total_params = sum(p.numel() for p in self.net.parameters())
        print(pytorch_total_params)
        
        
        self.net.train(False)
        self.net.cuda()

    def save(self, outpath):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                mask = mask.squeeze(0).cuda().float()
                resize_shape = (image.size(2), image.size(3))
                
                ###### Origin
                out5 ,out4, out3, out2, out1 = self.net(image)
                out = out1
                out = F.interpolate(out, size=shape, mode='bilinear')
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head  = outpath #os.path.join(outpath, self.cfg.datapath.split('\\')[-2])
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(os.path.join(head, name[0] +'.png'), np.round(pred))


                ### For Each Stage
                # out5 ,out4, out3, out2, out1 = self.net(image)
                # for i, output in enumerate([out5, out4, out3, out2, out1]):
                #     out = output
                #     out = F.interpolate(out, size=shape, mode='bilinear')
                #     pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                #     head  = os.path.join(outpath, f'stage{i+1}')
                #     if not os.path.exists(head):
                #         os.makedirs(head)
                #     cv2.imwrite(os.path.join(head, name[0]+'.png'), np.round(pred))



if __name__=='__main__':

    for path in ['../Dataset/Old_Camouflage/TestDataset/CAMO', '../Dataset/Old_Camouflage/TestDataset/CHAMELEON'
              ,'../Dataset/Old_Camouflage/TestDataset/COD10K', '../Dataset/Old_Camouflage/TestDataset/NC4K']:


        parser = argparse.ArgumentParser()

        parser.add_argument('--model_path',default= './models/PINet/model-64')
        parser.add_argument('--output_path',default='./Results/PINet' )
        args = parser.parse_args()

        datapath = path    #args.data_path
        modelpath = args.model_path
        

        ### Origin
        outpath = os.path.join(args.output_path, path.split('/')[-1])
        t = Test(dataset, PINet, datapath, modelpath)
        t.save(outpath)