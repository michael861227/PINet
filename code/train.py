#!/usr/bin/python3
#coding=utf-8

import os
import sys
import datetime
from types import DynamicClassAttribute

from torch._C import dtype
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset

from apex import amp
from network import PINet
from smoothnessloss import smoothness_loss


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce+wiou#.mean()

def structure_loss_sal(pred, mask, flag):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wbce + wiou  # .mean()
    

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        
def train(Dataset, Network):
    ## dataset

    cfg = Dataset.Config(datapath='../Dataset/Old_Camouflage/TrainDataset/',
                         savepath = './models/PINet',
                         mode='train_cam',
                         batch=36, lr=0.01, momen=0.9, decay=5e-4, epoch=64, clip=0.5, decay_rate=0.1, decay_epoch=50)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True)
    
    ## Network
    net    = Network()
    
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
    net.train(True)
    net.cuda()
    
    ## parameter
    base, head = [], []
    bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')
    mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)

    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            continue
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
            
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    params = net.parameters()

    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0
    smoth_loss = smoothness_loss(size_average=True)
    
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        #adjust_lr(optimizer, cfg.lr, epoch, cfg.decay_rate, cfg.decay_epoch)
        
        ### Origin
        for step, (image, mask, corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image) in enumerate(loader):
            image, mask, corner_mask, corner_mask_mid, corner_mask_hard, corner_mask_hardmid, flag, flag_mid, flag_hard, flag_hardmid, g_image = \
                image.cuda().float(), mask.cuda().float(), corner_mask.cuda().float(),corner_mask_mid.cuda().float(), \
                corner_mask_hard.cuda().float(), corner_mask_hardmid.cuda().float(), flag.cuda().float(), flag_mid.cuda().float(), \
                flag_hard.cuda().float(), flag_hardmid.cuda().float(), g_image.cuda().float()


            out4, out3, out2, out1, out = net(image)

            
            sloss = smoth_loss(out.sigmoid(), g_image) * 1.0
            loss1v = structure_loss(out, mask)
            loss1 = structure_loss(out1, mask)

            weight = torch.ones_like(mask)


            ### Origin
            loss4 = bceloss(out4, mask) * ((0.2 * corner_mask)+weight)
            loss3 = bceloss(out3, mask) * ((0.2 * corner_mask)+(0.2 * corner_mask_mid)+weight)
            loss2 = bceloss(out2, mask) * ((0.2 * corner_mask_hardmid)+(0.2 * corner_mask_mid)+(0.2 * corner_mask)+weight)
            
            
            ### Loss
            loss1v = loss1v.mean()
            loss1  = loss1.mean() 
            loss4 = loss4.mean() 
            loss3 = loss3.mean() 
            loss2 = loss2.mean() 
            
            loss = loss1v + sloss + loss1/2+loss4/12+loss3/6+loss2/3 


            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            
            optimizer.step()
            
            
            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1':loss1.item(), 'loss4':loss4.item(), 'loss3':loss3.item(), 'loss2':loss2.item()}, global_step=global_step)

            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))


        if epoch+1 == cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, PINet)
