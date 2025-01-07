import torch
import torch.nn as nn
import torch.nn.functional as F
from common_blocks import down_block, up_block

class convnet1(nn.Module):
    def __init__(self, in_ch=3, n_channels=8):
        super(convnet1, self).__init__()
        self.activation = F.relu
        
        self.conv1 = nn.Conv2d(in_ch, n_channels, kernel_size= 1, padding=0)
        self.conv2 = nn.Conv2d(n_channels, 1, kernel_size= 1, padding=0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        
        return out

if __name__ == "__main__":
    unet = convnet1()
    
    x = torch.rand(2, 3, 32, 32)
    out = unet(x)
    print("Output", out.shape)
