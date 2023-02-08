# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 20:07:59 2021

@author: wangm
"""

import torch
from torch import nn
from torch import optim
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.functional import fold, unfold
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from asteroid.masknn.recurrent import DPRNNBlock

from sklearn import metrics
from loss_set import AMSoftmax,LargeMarginSoftmaxV1

from torchsummaryX import summary

import config


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class dprnn_model(pl.LightningModule):

    def __init__(self,
        bn_chan=64,
        hid_size=64,
        chunk_size=45,
        hop_size=None,
        n_repeats=4,
        ):
        super(dprnn_model, self).__init__()
        
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats

        self.conv1 = nn.Conv1d(in_channels=1,out_channels=bn_chan, stride=4, kernel_size=8,padding=4)

        # Succession of DPRNNBlocks.
        net = []
        for x in range(self.n_repeats):
            net += [
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    dropout=0.5,
                )
            ]
        self.net = nn.Sequential(*net)
        
        self.fc1 = nn.Linear(self.bn_chan*4, self.bn_chan*4)
        self.fc2 = nn.Linear(self.bn_chan*4, 4)
        
        self.criteria = LargeMarginSoftmaxV1(reduction='mean')
        
    def forward(self, x):
        
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        x = x.unsqueeze(1)
        x = self.conv1(x)
        batch, n_filters, n_frames = x.size()
        output = unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output = self.net(output)
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        output = output.reshape(batch, self.bn_chan, -1)
        output = F.adaptive_avg_pool2d(output, (self.bn_chan,4))
        output = output.view(output.size(0), -1)
        
        output = F.relu(self.fc1(output))
        output = F.dropout(output, 0.5)
        output = self.fc2(output)
        
        return output
    
    
    # process inside the training loop
    def training_step(self, batch, batch_idx):

        x , y = batch
        x, y_a, y_b, lam = mixup_data(x,y)
        y_a = y_a.squeeze(1)
        y_b = y_b.squeeze(1)
        logits = self.forward(x)
        
        loss = lam * self.criteria(logits, y_a) + (1 - lam) * self.criteria(logits, y_b)
        
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    
    # process inside the validation loop
    def validation_step(self, batch, batch_idx):
        x , y = batch
        y = y.squeeze(1)
        logits = self.forward(x)
        val_loss = nn.CrossEntropyLoss()(logits, y)

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
    
        return {'logits': logits,
                'y': y
                }    

    def validation_epoch_end(self, outputs):

        
        logits_all = torch.vstack([x['logits'] for x in outputs])
        y_all = torch.hstack([x['y'] for x in outputs])
        
        logits_all = logits_all.argmax(1)
        
        logits_all = logits_all.cpu().numpy()
        y_all = y_all.cpu().numpy()
        
        #f1_score = f1(logits_all, y_all, num_classes=4, average='macro')
        target_names = ['A', 'N', 'O', 'G']
        print(metrics.classification_report(y_all, logits_all, target_names=target_names))
        score_dict = metrics.classification_report(y_all, logits_all, target_names=target_names, output_dict=True)
        f1_score = (score_dict['A']['f1-score']+score_dict['N']['f1-score']+score_dict['O']['f1-score'])/3.0
        #f1_score = metrics.f1_score(y_all, logits_all, average='macro')

       

        self.log('val_f1_score',f1_score)
  
    def configure_optimizers(self):
        
        if config.optim == 'Adam':
           optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        elif config.optim == 'AdamW':    
           optimizer = optim.AdamW(lr=1e-3, params=self.parameters())
               
        if config.scheduler is None:
            return optimizer
        else:       
            return {
                    'optimizer': optimizer,
                    #'lr_scheduler': ReduceLROnPlateau(
                    #    optimizer, mode='min', factor=config.scheduler_factor, patience=config.patience, verbose=True, min_lr=1e-5),
                    'lr_scheduler': ExponentialLR(optimizer, gamma=0.92),
                    'monitor': "val_loss"
                    }    
