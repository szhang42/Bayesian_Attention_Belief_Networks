# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        # Gumbel Softmax
        # noise = torch.rand(att.size()).cuda()
        # eps =1e-7
        # noise = noise.add_(eps).log_().neg_()
        # temperature =self.concrete_temp
        # att = (att+noise) / temperature
        # att = F.softmax(att, dim=1)

        att = F.softmax(att, dim=1)

        # Hard Attention
        # att = F.softmax(att, dim=1) # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
        # #print('att_map shape', att_map.shape)
        # # att_map = self.dropout(att_map)
        # att = torch.distributions.OneHotCategorical(att)
        # att = att.sample()
        # print('att_map_one', att_map_one)
        # log_probs = att_map_dis.log_prob(att_map_one)
        # self.log_probs = log_probs

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)




        # self.SA1_linear=nn.Linear(512, 512)
        # self.SA2_linear = nn.Linear(512, 512)
        # self.SGA1_linear=nn.Linear(512, 512)
        # self.SGA2_linear = nn.Linear(512, 512)
        # self.SGA3_linear = nn.Linear(512, 512)
        # self.SGA4_linear = nn.Linear(512, 512)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # print('la_mask', lang_feat_mask.shape)
        # print('image_mask', img_feat_mask.shape)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        d_k = lang_feat.size(-1)



        # lang_featSA1 = self.SA1_linear(lang_feat)
        # lang_featSA2 = self.SA2_linear(lang_feat)

        lang_featSA1 = lang_feat
        lang_featSA2 = lang_feat

        att_map_last = torch.matmul(
            lang_featSA1, lang_featSA2.transpose(-2, -1)
        ) / math.sqrt(d_k)







        # att_map_last = torch.matmul(
        #     lang_feat, lang_feat.transpose(-2, -1)
        # ) / math.sqrt(d_k)

        att_map_last_SA =att_map_last.unsqueeze(1)

        att_map_last_SA = att_map_last_SA.repeat(1, 8, 1, 1)

        # att_map_last_SA_mid = (torch.softmax(att_map_last_SA, dim=-1)).data.cuda()

        # att_map_last_SA_mid = (torch.softmax(att_map_last_SA, dim=-1)).cuda()

        shape = list(att_map_last_SA.shape)
        # att_map_last_SA_mid = (torch.from_numpy(np.zeros([shape[0], shape[1], shape[2], shape[3]]))).type(torch.float32).cuda()
        att_map_last_SA_mid = (torch.from_numpy(np.zeros([shape[0], shape[1], shape[2], shape[3]]))).type(torch.float32)



        hidden_SA = att_map_last_SA_mid
        # samp_wei_SA = att_map_last_SA_mid
        samp_wei_SA = 0

        d_k = img_feat.size(-1)



        # img_featSGA1 = self.SGA1_linear(img_feat)
        # img_featSGA2 = self.SGA2_linear(img_feat)

        img_featSGA1 = img_feat
        img_featSGA2 = img_feat
        att_map_last = torch.matmul(
            img_featSGA1, img_featSGA2.transpose(-2, -1)
        ) / math.sqrt(d_k)







        # att_map_last = torch.matmul(
        #     img_feat, img_feat.transpose(-2, -1)
        # ) / math.sqrt(d_k)


        att_map_last_SGA_1 =att_map_last.unsqueeze(1)

        att_map_last_SGA_1 = att_map_last_SGA_1.repeat(1, 8, 1, 1)

        # att_map_last_SGA_1_mid = (torch.softmax(att_map_last_SGA_1, dim=-1)).data.cuda()
        # att_map_last_SGA_1_mid = (torch.softmax(att_map_last_SGA_1, dim=-1)).cuda()



        shape2 = list(att_map_last_SGA_1.shape)
        # att_map_last_SGA_1_mid = (torch.from_numpy(np.zeros([shape2[0], shape2[1], shape2[2], shape2[3]]))).type(torch.float32).cuda()
        att_map_last_SGA_1_mid = (torch.from_numpy(np.zeros([shape2[0], shape2[1], shape2[2], shape2[3]]))).type(torch.float32)


        hidden_SGA_1 = att_map_last_SGA_1_mid
        # samp_wei_SGA_1 = att_map_last_SGA_1_mid
        samp_wei_SGA_1 = 0


        # print('SGA_last_1', (att_map_last_SGA_1 < 0).sum())

        d_k = img_feat.size(-1)





        # img_featSGA3 = self.SGA3_linear(img_feat)
        # lang_featSGA4 = self.SGA4_linear(lang_feat)


        img_featSGA3 = img_feat
        lang_featSGA4 = lang_feat
        att_map_last = torch.matmul(
            img_featSGA3, lang_featSGA4.transpose(-2, -1)
        ) / math.sqrt(d_k)








        # att_map_last = torch.matmul(
        #     img_feat, lang_feat.transpose(-2, -1)
        # ) / math.sqrt(d_k)


        att_map_last_SGA_2 = att_map_last.unsqueeze(1)

        att_map_last_SGA_2 = att_map_last_SGA_2.repeat(1, 8, 1, 1)

        # att_map_last_SGA_2_mid = (torch.softmax(att_map_last_SGA_2, dim=-1)).data.cuda()
        # att_map_last_SGA_2_mid = (torch.softmax(att_map_last_SGA_2, dim=-1)).cuda()





        shape3 = list(att_map_last_SGA_2.shape)
        # att_map_last_SGA_2_mid = (torch.from_numpy(np.zeros([shape3[0], shape3[1], shape3[2], shape3[3]]))).type(torch.float32).cuda()
        att_map_last_SGA_2_mid = (torch.from_numpy(np.zeros([shape3[0], shape3[1], shape3[2], shape3[3]]))).type(torch.float32)
        hidden_SGA_2 = att_map_last_SGA_2_mid
        # samp_wei_SGA_2 = att_map_last_SGA_2_mid
        samp_wei_SGA_2 = 0

        h_SA = hidden_SA
        h_SGA_1 = hidden_SGA_1
        h_SGA_2 = hidden_SGA_2

        # h_SA = torch.tensor(0.0).type(torch.float32)
        # h_SGA_1 = torch.tensor(0.0).type(torch.float32)
        # h_SGA_2 = torch.tensor(0.0).type(torch.float32)

        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            att_map_last_SA_mid,
            att_map_last_SGA_1_mid,
            att_map_last_SGA_2_mid,
            h_SA,
            h_SGA_1,
            h_SGA_2,
            samp_wei_SA,
            samp_wei_SGA_1,
            samp_wei_SGA_2
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
