from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
import sys
# sys.path.insert(0,'/PointFlow/')
sys.path.insert(0,'/PointAttN/')
from utils.model_utils import *

from utils.mm3d_pn2 import furthest_point_sample, gather_points




# class MLP_Res(nn.Module):
class Dynamic_confidence_filter(nn.Module):    
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(Dynamic_confidence_filter, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.relu = nn.GELU()
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)
        self.l2_loss = nn.MSELoss()

    def forward(self, fine, gt_points, x2_d, epoch, is_training):
        """
        Args:
            x: (B, out_dim, n)
        """

        idx = furthest_point_sample(gt_points.contiguous(), fine.size(-1))
        gt_points_sampled = gather_points(gt_points.transpose(1, 2).contiguous(), idx)
        
        fine = fine.transpose(1, 2).contiguous()
        gt_points_sampled = gt_points_sampled.transpose(1, 2).contiguous()

        dist1, dist2, idx1, idx2=calc_all_dist(fine,gt_points_sampled)

        confidence_score=torch.exp(-dist1)

        confidence_score_predict = self.conv_out3(self.relu(self.conv_out2(self.relu(self.conv_out1(x2_d)))))

        confidence_score_loss=self.l2_loss(confidence_score_predict.squeeze(),confidence_score)

        if epoch<50:
            alpha1 = 0.01
            alpha2 = 0.02

            # print(filtered_fine.shape)
            # exit()


            # if is_training==True:
            #     confidence_score=confidence_score
            # else:
            #     confidence_score=confidence_score_predict   
            # # confidence_score=confidence_score.unsqueeze(-1)
            # merge_middle_1=torch.cat((fine,confidence_score.unsqueeze(-1)),-1)
            # merge_middle_1_indices= merge_middle_1[:, :, -1].sort()[1]
            # merge_middle_1_list=[]
            # for i in range(fine.size(0)):
            #     merge_middle_1_list.append([i])
            # merge_middle_1=merge_middle_1[merge_middle_1_list, merge_middle_1_indices]
            # # merge_middle_1=merge_middle_1[[[0], [1], [2]], merge_middle_1_indices]
            # merge_middle_1=merge_middle_1[:,:,0:-1]
            # # merge_middle_1=merge_middle_1[:,128:,:]
            # filtered_fine=merge_middle_1[:,128:,:]
            # print(merge_middle_1.shape)
            # exit()

            filtered_fine=fine
        elif epoch<100:
            alpha1 = 0.05
            alpha2 = 0.1
            if is_training==True:
                confidence_score=confidence_score
            else:
                confidence_score=confidence_score_predict   
            # merge_middle_1=torch.cat((fine,confidence_score.squeeze().unsqueeze(-1)),-1)
            # merge_middle_1_indices= merge_middle_1[:, :, -1].sort()[1]
            # merge_middle_1_list=[]
            # for i in range(fine.size(0)):
            #     merge_middle_1_list.append([i])
            # merge_middle_1=merge_middle_1[merge_middle_1_list, merge_middle_1_indices]
            # # merge_middle_1=merge_middle_1[[[0], [1], [2]], merge_middle_1_indices]
            # merge_middle_1=merge_middle_1[:,:,0:-1]
            # # merge_middle_1=merge_middle_1[:,128:,:]
            # filtered_fine=merge_middle_1[:,128:,:]
            filtered_fine, confidence_score = sort_pc_according_to_conf(fine, confidence_score)

        else:
            alpha1 = 0.1
            alpha2 = 0.2
            if is_training==True:
                confidence_score=confidence_score
            else:
                confidence_score=confidence_score_predict   
            # merge_middle_1=torch.cat((fine,confidence_score.squeeze().unsqueeze(-1)),-1)
            # merge_middle_1_indices= merge_middle_1[:, :, -1].sort()[1]
            # merge_middle_1_list=[]
            # for i in range(fine.size(0)):
            #     merge_middle_1_list.append([i])
            # merge_middle_1=merge_middle_1[merge_middle_1_list, merge_middle_1_indices]
            # # merge_middle_1=merge_middle_1[[[0], [1], [2]], merge_middle_1_indices]
            # merge_middle_1=merge_middle_1[:,:,0:-1]
            # # merge_middle_1=merge_middle_1[:,128:,:]
            # filtered_fine=merge_middle_1[:,196:,:]
            filtered_fine, confidence_score = sort_pc_according_to_conf(fine, confidence_score)
        return filtered_fine, confidence_score_loss

def sort_pc_according_to_conf(fine_pc, conf_score):
    """
        fine_pc - sample shape: torch.randn(batch_sz, 5, 3)
        conf_score - sample shape = torch.randn(batch_sz, 5)

        return: return shapes same as input shapes
    """
    fine_pc = fine_pc.cuda()
    conf_score = conf_score.cuda()

    fine_sort = torch.FloatTensor(*fine_pc.shape).cuda()

    for x in range(fine_pc.shape[0]):
        fine_sort[x] = fine_pc[x][conf_score[x].argsort(descending=True)]

    conf_score = torch.sort(conf_score, descending=True)[0]

    return fine_sort, conf_score

class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)
        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse,feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

        y_up = y.repeat(1,1,self.ratio)
        y_cat = torch.cat([y3,y_up],dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

        return x, y3

class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)

        self.filtered_corse_pc = Dynamic_confidence_filter(in_dim=256, hidden_dim=128, out_dim=128)

    # def forward(self, points):
    def forward(self, points, gt_points, epoch, is_training):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1,x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()
        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        filtered_fine, confidence_score_loss=self.filtered_corse_pc(fine,gt_points,x2_d,epoch,is_training) 
        filtered_fine=filtered_fine.transpose(1, 2)
        # print(fine.shape)
        # print(filtered_fine.shape)
        # exit()

        # return x_g, fine
        return x_g, filtered_fine, confidence_score_loss


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        else:
            ValueError('dataset is not exist')

        self.encoder = PCT_encoder()

        self.refine = PCT_refine(ratio=step1)
        self.refine1 = PCT_refine(ratio=step2)


    # def forward(self, x, gt=None, is_training=True):
    def forward(self, x, gt=None, epoch=None, is_training=True):
        # print(gt.shape)
        # print(epoch)
        # exit()
        # feat_g, coarse = self.encoder(x)
        feat_g, coarse, confidence_score_loss = self.encoder(x,gt,epoch,is_training)
        # print(confidence_score_loss)
        new_x = torch.cat([x,coarse],dim=2)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        if is_training:
            loss3, _ = calc_cd(fine1, gt)
            gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()

            loss2, _ = calc_cd(fine, gt_fine1)
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()

            loss1, _ = calc_cd(coarse, gt_coarse)

            # confidence_score_loss
            # total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean() + confidence_score_loss
            # print(total_train_loss)
            # exit()
            return fine, loss2, total_train_loss
        else:
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)

            return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}

