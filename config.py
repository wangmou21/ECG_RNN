# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 20:04:43 2021

@author: wangm
"""

import numpy as np
import torch
weights = np.array([6.7, 1, 2.1, 1])
weights = torch.from_numpy(weights.astype(np.float32)).cuda()


# Maximum sequence length
max_length = 18286

sample_rate = 300 # 300Hz

split_seed = 10

segment = 20

model_type = 'dprnn_ek8_ec64_hid64_r4_fc512_lm_0.3_len4096_dr0.5_mixup' 

# train setting
early_stop = True
patience_stop = 15                  # 训练提前终止的epoch数
gpus = [2]            # 所用到的GPU的编号
distributed_backend = "dp"         # 可选ddp, dp 多卡时建议ddp
epochs_max = 100                   # 最大训练的epoch的次数
gradient_clipping = 5.0             # 学习率裁剪的比例 

# optimizer
optim = 'AdamW'
#optim = 'Adam'
scheduler = True                    # True: ReduceLROnPlateau, None: 
learning_rate = 1e-3                # 初始学习率
patience = 5                  # 学习率裁剪的epoch数
scheduler_factor = 0.5

num_shift = np.array([150, 1142, 560, 350])
seg_len = 4096
