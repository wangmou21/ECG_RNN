# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:07:18 2021

@author: wangm
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


from model import dprnn_model
import config

def training_k_fold():
    
    # Data folder and hdf5 dataset file
    data_root = os.path.normpath('..')
    hd_file = os.path.join(data_root, 'physio.h5')
    label_file = os.path.join(data_root, 'REFERENCE-v3.csv')
    
    # Open hdf5 file
    h5file =  h5py.File(hd_file, 'r')
    
    # Get a list of dataset names 
    dataset_list = list(h5file.keys())
    
    # Load the labels
    label_df = pd.read_csv(label_file, header = None, names = ['name', 'label'])
    # Filter the labels that are in the small demo set
    label_df = label_df[label_df['name'].isin(dataset_list)]
    
    # Encode labels to integer numbers
    label_set = list(sorted(label_df.label.unique()))
    encoder = LabelEncoder().fit(label_set)
    label_set_codings = encoder.transform(label_set)
    label_df = label_df.assign(encoded = encoder.transform(label_df.label))
    
    print('Unique labels:', encoder.inverse_transform(label_set_codings))
    print('Unique codings:', label_set_codings)
    #print('Dataset labels:\n', label_df.iloc[100:110,])
    
    n_classes = len(label_df.label.unique())    
    
    ##
    label_all = np.array(label_df['encoded'])
    label_all = label_all[:,None]
    data_all = [ h5file[i]['ecgdata'][:, 0] for i in dataset_list ]
    len_all = [ len(h5file[i]['ecgdata'][:, 0]) for i in dataset_list ]
        
    split_seed = config.split_seed      
    print("split_seed: ", split_seed)

    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    fold = 0
    for train_index, val_index in skf.split(data_all, label_all):
        # Store the ids and labels in dictionaries
        print("Fold : ", fold)
        X_train = [data_all[i] for i in train_index]
        X_val = [data_all[i] for i in val_index]
        Y_train = [label_all[i] for i in train_index]
        Y_val = [label_all[i] for i in val_index]
        
        print("Y_train len: ", len(Y_train), "Y_val len:", len(Y_val))
        unique, count = np.unique(Y_train, return_counts = True)
        print("Y_train values: ", unique, "count: ", count)
        unique, count = np.unique(Y_val, return_counts = True)
        print("y_val values: ", unique, "count: ", count)
        print("\n")


        train_set = ECGDataset(X_train, Y_train, segment=config.segment)
        val_set = ECGDataset(X_val, Y_val, segment=None)
        
        train_loader = DataLoader(dataset=train_set,batch_size=32,shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=False, num_workers=4)
        
        model = dprnn_model()
        
        ckpt_cb = ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            dirpath='./space/checkpoints/'+config.model_type+'/'+str(fold), 
            filename='best',
            save_last=False,
            )
        
        es = EarlyStopping(
            monitor='val_loss', 
            patience=config.patience_stop, 
            mode='min',
            )
        
        Logger = TensorBoardLogger(
            save_dir='./space/logs/', 
            name=config.model_type+str(fold),
            )
        
        Callbacks = [es, ckpt_cb]
        
        trainer = pl.Trainer(
            max_epochs=config.epochs_max,
            gpus=config.gpus, 
            #precision=16,
            callbacks=Callbacks,
            logger=Logger,
            #distributed_backend=config.distributed_backend,
            num_sanity_val_steps=0,
            # fast_dev_run=True
            )
        
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        
        fold += 1


class ECGDataset(Dataset):

    def __init__(self, X, Y, segment):
        self.X = X
        self.Y = Y

        self.seg_len = None if segment is None else int(segment * config.sample_rate)
        self.EPS = 1e-8
        self.like_test = self.seg_len is None

    def __getitem__(self, index):
        label = self.Y[index]
        data_temp = self.X[index]
        
        if not self.like_test:
            if self.seg_len//len(data_temp)==1:
                data_temp = np.hstack((data_temp, data_temp[0:self.seg_len-len(data_temp)]))
            elif self.seg_len//len(data_temp)==2:
                data_temp = np.hstack((data_temp, data_temp, data_temp[0:self.seg_len-2*len(data_temp)]))
        
        if len(data_temp) == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, len(data_temp) - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        
        inputs = data_temp[rand_start:stop]
        
        inputs = torch.from_numpy(inputs.astype(np.float32))
        label = torch.from_numpy(label.astype(np.compat.long))
        return inputs, label

    def __len__(self):
        return len(self.X)   
    
    
if __name__ == "__main__":
    
    print("Adaptor Training K Fold\n")       
    training_k_fold()
