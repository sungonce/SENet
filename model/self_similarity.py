# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch.nn as nn
import torch.nn.functional as F


class SSM(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size=7, ksize=3):
        super(SSM, self).__init__()
        
        self.ch_reduction_encoder = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False, padding=0)
        self.SCC = SelfCorrelationComputation(unfold_size=unfold_size)
        self.SSE = SelfSimilarityEncoder(in_ch, mid_ch, unfold_size=unfold_size, ksize=ksize)

        self.FFN = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True, padding=0))
     
    def forward(self, ssm_input_feat):
        q = self.ch_reduction_encoder(ssm_input_feat)
        q = F.normalize(q, dim=1, p=2)
            
        self_sim = self.SCC(q)
        self_sim_feat = self.SSE(self_sim)
        ssm_output_feat = ssm_input_feat + self_sim_feat
        ssm_output_feat = self.FFN(ssm_output_feat)

        return ssm_output_feat

class SelfCorrelationComputation(nn.Module):
    def __init__(self, unfold_size=5):
        super(SelfCorrelationComputation, self).__init__()
        self.unfold_size = (unfold_size, unfold_size)
        self.padding_size = unfold_size // 2
        self.unfold = nn.Unfold(kernel_size=self.unfold_size, padding=self.padding_size)

    def forward(self, q):
        b, c, h, w = q.shape

        q_unfold = self.unfold(q)  # b, cuv, h, w
        q_unfold = q_unfold.view(b, c, self.unfold_size[0], self.unfold_size[1], h, w) # b, c, u, v, h, w
        self_sim = q_unfold * q.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        self_sim = self_sim.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v

        return self_sim.clamp(min=0)
        
class SelfSimilarityEncoder(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size, ksize):
        super(SelfSimilarityEncoder, self).__init__()
            
        def make_building_conv_block(in_channel, out_channel, ksize, padding=(0,0,0), stride=(1,1,1), bias=True, conv_group=1):
            building_block_layers = []
            building_block_layers.append(nn.Conv3d(in_channel, out_channel, (1, ksize, ksize),
                                             stride=stride, bias=bias, groups=conv_group, padding=padding))
            building_block_layers.append(nn.BatchNorm3d(out_channel))
            building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        conv_in_block_list = [make_building_conv_block(mid_ch, mid_ch, ksize) for _ in range(unfold_size//2)]
        self.conv_in = nn.Sequential(*conv_in_block_list)
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=True, padding=0),
            nn.BatchNorm2d(in_ch))

    def forward(self, x):
        b, c, h, w, u, v = x.shape

        x = x.view(b, c, h * w, u, v)
        x = self.conv_in(x)
        c = x.shape[1]
        x = x.mean(dim=[-1,-2]).view(b, c, h, w)
        x = self.conv1x1_out(x)  # [B, C3, H, W] -> [B, C4, H, W]

        return x
