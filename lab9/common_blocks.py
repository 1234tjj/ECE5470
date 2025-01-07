import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout_prob=0, max_pooling=True):
        super(down_block, self).__init__()
        
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = F.relu
        
        # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
        if dropout_prob > 0:
            self.dropout = nn.Dropout2d(p=dropout_prob)

        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        if max_pooling:
            self.maxpool = nn.MaxPool2d(2)

    def forward(self, inputs):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        conv = self.activation(conv)
        
        if self.dropout_prob > 0:
            conv = self.dropout(conv)
            
        next_layer = conv
        skip_connection = conv
        
        if self.max_pooling:
            next_layer = self.maxpool(conv)
        
        return next_layer, skip_connection
   
class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(up_block, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = F.relu
        
    def forward(self, expansive_input, contractive_input=None):
        up = self.up(expansive_input)
        
        merge = torch.cat([up, contractive_input], axis=1)
        
        out = self.conv(merge)
        out = self.bn(out)
        out = self.activation(out)
        return out   
 
