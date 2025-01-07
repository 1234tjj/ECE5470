import torch
import torch.nn as nn
import torch.nn.functional as F
from common_blocks import down_block, up_block

class unet2(nn.Module):
    def __init__(self, in_ch=3, n_channels=8):
        super(unet2, self).__init__()
        self.activation = F.relu
        
        self.down1 = down_block(in_ch, n_channels)

        self.bridge = down_block(in_ch=n_channels, out_ch=n_channels*2, max_pooling=False)

        self.up1 = up_block(in_ch=n_channels*2, out_ch=n_channels)
        self.final_conv = nn.Conv2d(n_channels, 1, kernel_size= 1, padding=0)
    
    def forward(self, x):
        out, skip1 = self.down1(x)
        out, _ = self.bridge(out)
        
        out = self.up1(out, skip1)
        out = self.final_conv(out)
        
        return out

if __name__ == "__main__":
    unet = unet2()
    
    x = torch.rand(2, 3, 32, 32)
    out = unet(x)
    print("Output", out.shape)
