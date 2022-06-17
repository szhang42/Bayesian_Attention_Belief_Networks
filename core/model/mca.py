# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import numpy as np

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, HP, in_features, out_features, prior_features):
        super(MHAtt, self).__init__()
        self.HP = HP
        self.linear_v = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_k = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_k2 = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_q = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_q2 = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_prior = nn.Linear(prior_features, prior_features)
        # self.concrete_temp = 0.1
        #TODO: prefer to have it in config.
        self.forward_mode_att = True
        self.dropout = nn.Dropout(HP.DROPOUT_R)
        self.k_weibull = self.HP.k_weibull

        if self.HP.att_kl != 0.0 and self.HP.att_type == 'soft_weibull':
            if self.HP.att_prior_type == 'parameter':
                self.alpha_gamma = nn.Parameter(torch.Tensor(1))
                self.alpha_gamma.data.fill_(self.HP.alpha_gamma)
                self.beta_gamma = torch.tensor(self.HP.beta_gamma).type(torch.float32)
            else:
                self.alpha_gamma = torch.tensor(self.HP.alpha_gamma).type(torch.float32)
                self.beta_gamma = torch.tensor(self.HP.beta_gamma).type(torch.float32)

        # if self.HP.att_kl != 0.0 and self.HP.att_type == 'gamma_att':
        if self.HP.att_type == 'gamma_att':
            if self.HP.model_type == 'whai':

                self.w_3 = nn.Parameter(torch.Tensor(in_features, out_features))
                self.w_3.data.normal_(0.01, 0.1)
                # self.w_3.data.normal_(0.001, 0.001)

                self.b_3 = nn.Parameter(torch.Tensor(out_features, 1))
                self.b_3.data.normal_(0.001, 0.001)
                # self.b_3.data.normal_(0.01, 0.01)

                self.w_2 = nn.Parameter(torch.Tensor(in_features, out_features))
                self.w_2.data.normal_(0.01, 0.1)
                # self.w_2.data.normal_(0.001, 0.001)

                self.b_2 = nn.Parameter(torch.Tensor(out_features, 1))
                self.b_2.data.normal_(0.001, 0.001)
                # self.b_2.data.normal_(0.01, 0.01)


                self.w_1 = nn.Parameter(torch.Tensor(in_features, out_features))
                self.w_1.data.normal_(0.01, 0.1)
                # self.w_1.data.normal_(0.001, 0.001)

                self.b_1 = nn.Parameter(torch.Tensor(out_features, 1))
                self.b_1.data.normal_(0.001, 0.001)
                # self.b_1.data.normal_(0.01, 0.01)

                if self.HP.att_contextual_se:
                    self.se_linear1 = nn.Linear(self.HP.HIDDEN_SIZE_HEAD, self.HP.att_se_hid_size)
                    self.se_linear2 = nn.Linear(self.HP.att_se_hid_size, 1)
                    if self.HP.att_se_nonlinear == 'lrelu':
                        self.se_nonlinear = nn.LeakyReLU(0.3)  # TODO: tune.
                    elif self.HP.att_se_nonlinear == 'relu':
                        self.se_nonlinear = nn.ReLU()
                    elif self.HP.att_se_nonlinear == 'sigmoid':
                        self.se_nonlinear = nn.Sigmoid()
                    elif self.HP.att_se_nonlinear == 'tanh':
                        self.se_nonlinear = nn.Tanh()
                    self.contextual_prior_net = nn.Sequential(*[self.se_linear1, self.se_nonlinear, self.se_linear2])



            if self.HP.att_prior_type == 'parameter':
                self.alpha_gamma = nn.Parameter(torch.Tensor(1))
                self.alpha_gamma.data.fill_(self.HP.alpha_gamma)
                self.beta_gamma = torch.tensor(self.HP.beta_gamma).type(torch.float32)
                self.prior_gamma = torch.tensor(self.HP.prior_gamma).type(torch.float32)
            else:
                self.alpha_gamma = torch.tensor(self.HP.alpha_gamma).type(torch.float32)
                self.beta_gamma = torch.tensor(self.HP.beta_gamma).type(torch.float32)
                self.prior_gamma = torch.tensor(self.HP.prior_gamma).type(torch.float32)
                self.prior_scores = torch.tensor(self.HP.alpha_gamma).type(torch.float32)

        elif self.HP.att_type == 'soft_lognormal':
            self.sigma_normal_posterior = self.HP.sigma_normal_posterior
            if self.HP.att_kl != 0.0:
                if self.HP.att_prior_type == 'parameter':
                    self.sigma_normal_prior = nn.Parameter(torch.Tensor(1))
                    self.sigma_normal_prior.data.fill_(self.HP.sigma_normal_prior)
                else:
                    self.sigma_normal_prior = torch.tensor(self.HP.sigma_normal_prior).type(torch.float32)
                self.mean_normal_prior = torch.tensor(0.0).type(torch.float32)

        if self.HP.att_prior_type == 'contextual':
            if self.HP.att_contextual_se:
                self.se_linear1 = nn.Linear(self.HP.HIDDEN_SIZE_HEAD, self.HP.att_se_hid_size)
                self.se_linear2 = nn.Linear(self.HP.att_se_hid_size, 1)
                if self.HP.att_se_nonlinear == 'lrelu':
                    self.se_nonlinear = nn.LeakyReLU(0.3)  # TODO: tune.
                elif self.HP.att_se_nonlinear == 'relu':
                    self.se_nonlinear = nn.ReLU()
                elif self.HP.att_se_nonlinear == 'sigmoid':
                    self.se_nonlinear = nn.Sigmoid()
                elif self.HP.att_se_nonlinear == 'tanh':
                    self.se_nonlinear = nn.Tanh()
                self.contextual_prior_net = nn.Sequential(*[self.se_linear1, self.se_nonlinear, self.se_linear2])

                # def contextual_prior_net(self, x):
                #     return self.se_linear2(self.se_nonlinear(self.se_linear1(x)))
            else:
                self.contextual_prior_net = nn.Linear(self.HP.HIDDEN_SIZE_HEAD, 1)

    def forward(self, v, k, q, mask, att_map_last, hidden, samp_wei, laysa):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k1 = self.linear_k(k).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q1 = self.linear_q(q).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k2 = self.linear_k2(k).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q2 = self.linear_q2(q).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        if self.HP.model_type == 'bam_contexk':
            atted, k_wei= self.att(v, k1, q1, k2, q2, mask, att_map_last, hidden, samp_wei)
        if self.HP.model_type == 'whai':
            atted, hid, samp_wei = self.att(v, k1, q1, k2, q2, mask, att_map_last, hidden, samp_wei, laysa)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HP.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        # return atted
        if self.HP.model_type == 'bam_contexk':
            return atted, k_wei
        if self.HP.model_type == 'whai':
            return atted, hid, samp_wei

    def att(self, value, key, query, key2, query2, mask, att_map_last, hidden_pre, samp_wei_previous, laysa):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        # print('score_shape', scores.shape)

        scores2 = torch.matmul(
            query2, key2.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            scores2 = scores2.masked_fill(mask, -1e9)
        # TODO: combine HP.hard_gumbel and HP.Reinforce to HP.att_optim_type and use it as string.
        # TODO: do experiment to see the effect of dropout here at some point.
        ## Gumbel softmax
        if self.HP.att_type == 'hard_attention':
            if self.HP.att_optim_type == 'gumbel':
                if self.training:
                    noise = torch.rand(scores.size()).cuda()
                    eps = 1e-7
                    # noise = noise.add_(eps).log_().neg_()
                    noise = -(torch.log(-torch.log(noise + eps) + eps))
                    # TODO: double check whether the gumbel is correct.
                    # TODO: Temperature for gumbel has some effect, do experiment to see.
                    temperature = self.HP.concrete_temp
                    att_map = (scores + noise) / temperature
                    att_map_ori = F.softmax(att_map, dim=-1)
                    self.att_map_ori = att_map.data
                    att_map = self.dropout(att_map_ori)
                else:
                    att_map = F.softmax(scores, dim=-1)
            ## Hard attention with Reinforce
            if self.HP.att_optim_type == 'reinforce':
                if self.training:
                    eps = 1e-7
                    att_map_ori = F.softmax(scores, dim=-1)  # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
                    self.att_map_ori = scores.data
                    att_map_dis = torch.distributions.Categorical(att_map_ori)
                    att_map = att_map_dis.sample()
                    log_probs = torch.log(att_map_ori + eps)
                    # print('att_map', att_map.shape)
                    # print('log_prob1', log_probs.shape)
                    log_probs = log_probs.gather(3, att_map.unsqueeze(3))
                    # print('log_prob2', log_probs.shape)
                    # log_probs = att_map_dis.log_prob(att_map)
                    self.log_probs = log_probs
                    # TODO: remove the comments.
                else:
                    att_map = F.softmax(scores, dim=-1)

        ## variational attention with gumbel
        if self.HP.att_type == 'hard_var_att':
            if self.HP.att_optim_type == 'gumbel':
                if self.training:
                    eta_const = -1.38
                    # TODO: make eta learnable as in l-0 arm
                    shape = list(scores.shape)
                    self.att_eta = (
                                torch.from_numpy(np.ones([shape[0], shape[1], shape[2], shape[3]])) * eta_const).type(
                        torch.float32).cuda()
                    # self.att_eta = nn.Parameter(torch.Tensor(list(scores.shape)[3])).cuda()
                    # print('eta', self.att_eta.shape)
                    # self.k = 0.7
                    # TODO: why k?
                    noise = torch.rand(scores.size()).cuda()
                    eps = 1e-7
                    noise = -(torch.log(-torch.log(noise + eps) + eps))
                    temperature = self.HP.concrete_temp
                    # TODO: check gumbel
                    att_map_ori = F.softmax(scores, dim=-1)
                    self.att_map_ori = scores.data
                    att_map = (att_map_ori + noise) / temperature
                    att_map = F.softmax(att_map, dim=-1)  # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
                    self.att_post_nll_true = -(att_map * (torch.log(att_map_ori + eps)))
                    att_prior_pi = F.softmax(self.att_eta, dim=-1).type_as(self.att_post_nll_true)
                    # TODO: prior is wrong###
                    # print('att_map', att_map.shape)
                    # print('prior', att_prior_pi.shape)
                    self.att_prior_nll_true = -(att_map * torch.log(att_prior_pi + eps))
                else:
                    att_map = F.softmax(scores, dim=-1)

            # Variational attention with reinforce + baseline (soft attention)
            if self.HP.att_optim_type == 'reinforce_base':
                if self.forward_mode_att:
                    if self.training:
                        eta_const = -1.38
                        # print('scores', scores.shape)
                        shape = list(scores.shape)
                        self.att_eta = (
                                    torch.from_numpy(np.ones([shape[0], shape[1], shape[2], shape[3]])) * eta_const).type(
                            torch.float32).cuda()
                        # print('eta1', self.att_eta)
                        # self.att_eta = nn.Parameter(torch.Tensor(list(scores.shape)[3])).cuda()
                        # self.k = 0.01
                        eps = 1e-7
                        att_map_ori = F.softmax(scores, dim=-1)  # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
                        self.att_map_ori = scores.data
                        # print('1', att_map_ori.shape)
                        # att_map = self.dropout(att_map)
                        att_map_dis = torch.distributions.Categorical(att_map_ori)
                        att_map = att_map_dis.sample()
                        log_probs = torch.log(att_map_ori + eps)
                        log_probs = log_probs.gather(3, att_map.unsqueeze(3))
                        att_map = torch.distributions.OneHotCategorical(att_map_ori).sample()
                        # print('2', att_map.shape)
                        # print('att_map_one', att_map_one)
                        # log_probs = att_map_dis.log_prob(att_map)
                        # TODO: double check that log_prob is doing what you want, the gradient is not stopped.
                        self.log_probs = log_probs
                        self.att_post_nll_true = -(
                                    att_map * (torch.log(att_map_ori + eps)))  # negative likelihood for posterior
                        att_prior_pi = F.softmax(self.att_eta, dim=-1).type_as(self.att_post_nll_true)
                        # print('prior', att_prior_pi)
                        self.att_prior_nll_true = -(
                                    att_map * torch.log(att_prior_pi + eps))  # negative log likehood for prior
                    else:
                        att_map = F.softmax(scores, dim=-1)


                else:
                    eta_const = -1.38
                    shape = list(scores.shape)
                    self.att_eta = (
                                torch.from_numpy(np.ones([shape[0], shape[1], shape[2], shape[3]])) * eta_const).type(
                        torch.float32).cuda()
                    # self.att_eta = nn.Parameter(torch.Tensor(list(scores.shape)[3])).cuda()
                    eps = 1e-7
                    att_map_ori = F.softmax(scores, dim=-1)  # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
                    self.att_map_ori = scores.data
                    # att_prior_pi = F.softmax(self.att_eta, dim=-1)
                    # self.att_prior_nll_true = -(
                    #             att_prior_pi * (torch.log(att_prior_pi + eps)))  ### negative log likehood for prior
                    # self.att_post_nll_true = -(att_prior_pi * (torch.log(att_map_ori + eps))).type_as(
                    #     self.att_prior_nll_true)  # negative likelihood for posterior
                    self.att_prior_nll_true = 0
                    self.att_post_nll_true = 0

                    # TODO: att_prior_nll_true should be att_prior_nll, and it equals 0. same with the post.
                    # TODO: remove the redundant code here.
                    att_map = att_map_ori
        # else:
        #     att_map = F.softmax(scores, dim=-1)

        if self.HP.att_type == 'soft_attention':
            att_map_ori = F.softmax(scores, dim=-1)  # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
            self.att_map_ori = scores.data
            att_map = F.softmax(scores, dim=-1)


        if self.HP.att_type =='soft_weibull':
            # self.k_weibull = self.HP.k_weibull
            # self.alpha_gamma = self.HP.alpha_gamma
            # self.beta_gamma = self.HP.beta_gamma

            if self.HP.att_prior_type == 'contextual' and self.training:
                # self.alpha_gamma = torch.exp(self.contextual_prior_net(key)) * self.beta_gamma #TODO: value or key? tune.
                # between 0 and 1 sigmoid
                # self.alpha_gamma = torch.sigmoid(self.contextual_prior_net(key) * self.HP.optk) * self.beta_gamma

                #softmax
                # self.alpha_gamma = torch.softmax((self.contextual_prior_net(key) * self.HP.optk), dim=2) * self.beta_gamma

                self.alpha_gamma = torch.softmax((self.contextual_prior_net(key)),dim=2) * self.HP.optk * self.beta_gamma

                # between 0 and infty
                # alpha_sigmoid = torch.sigmoid(self.contextual_prior_net(key))
                # self.alpha_gamma = alpha_sigmoid / (1 - alpha_sigmoid) * self.beta_gamma

                #TODO: try different functions to convert it to postive numbers(relu, sqrt(sigmoid(x)/(1- sigmoid(x))))
                #TODO: may want to add KL in the beginning so that the prior can be trained.
                #TODO: add more complex things like apply squeeze and excite to channel dimension?
                self.alpha_gamma = self.alpha_gamma.transpose(2, 3)
                self.alpha_gamma.masked_fill(mask, 0)
                # print('alpha shape', self.alpha_gamma.shape)

            eps = 1e-20
            # scores =F.softmax(scores, dim=-1)
            # scores =torch.log(scores + eps)
            att_map_ori = F.log_softmax(scores, dim=-1)
            self.att_map_ori = scores.data
            scores =att_map_ori
            # print(scores.shape)
            if self.training:
                u_weibull = torch.rand_like(scores)
                # print(u_weibull.shape)
                lambda_weibull = torch.exp(scores) / torch.exp(torch.lgamma(1 + 1.0 / torch.tensor(self.k_weibull)).type_as(u_weibull))
                # print(lambda_weibull.shape)
                #sample_weibull  = lambda_weibull * torch.exp(1.0 / self.k_weibull * torch.log(- torch.log(1.0 - u_weibull + eps) + eps))
                sample_weibull = lambda_weibull * (-torch.log(1.0 - u_weibull + eps)) ** (1.0 / self.k_weibull)
                # print(sample_weibull.shape)
                att_map = sample_weibull / (sample_weibull.sum(-1, keepdim=True) +eps)
                if self.HP.att_kl != 0.0:
                    # print('alpha', self.alpha_gamma.shape)
                    # print('scores', scores.shape)
                    # KL = self.alpha_gamma * scores - np.euler_gamma * self.alpha_gamma / self.k_weibull \
                    #      - torch.log(torch.tensor(self.k_weibull) + eps) - self.beta_gamma * lambda_weibull * torch.exp(
                    #     torch.lgamma(1 + 1.0 / torch.tensor(self.k_weibull))) + np.euler_gamma + 1.0 + \
                    #      self.alpha_gamma * torch.log(torch.tensor(self.beta_gamma) + eps) - torch.lgamma(torch.tensor(self.alpha_gamma))
                    KL = self.alpha_gamma * scores - np.euler_gamma * self.alpha_gamma / self.k_weibull \
                         - torch.log(torch.tensor(self.k_weibull) + eps) - self.beta_gamma * lambda_weibull * torch.exp(
                        torch.lgamma(1 + 1.0 / torch.tensor(self.k_weibull))) + np.euler_gamma + 1.0 + \
                         self.alpha_gamma * torch.log(torch.tensor(self.beta_gamma) + eps) - torch.lgamma(torch.tensor(self.alpha_gamma) + eps)

                    # print('max', torch.max(self.alpha_gamma))
                    # print('min', torch.min(self.alpha_gamma))

                    # self.KL_backward = -KL.mean(-1).mean(0)
                    self.KL_backward =-KL.mean()
            else:
                att_map = F.softmax(scores, dim=-1)

        if self.HP.att_type == 'gamma_att':
            eps = 1e-20
            att_map_ori = F.log_softmax(scores, dim=-1)
            self.att_map_ori = scores.data
            scores = att_map_ori

            att_map_ori2 = F.log_softmax(scores2, dim=-1)
            self.att_map_ori2 = scores2.data
            scores2 = att_map_ori2

            if self.HP.att_prior_type == 'contextual' and self.training:
                self.alpha_gamma = torch.softmax((self.contextual_prior_net(key)),
                                                 dim=2) * self.HP.optk * self.beta_gamma

                # between 0 and infty
                # alpha_sigmoid = torch.sigmoid(self.contextual_prior_net(key))
                # self.alpha_gamma = alpha_sigmoid / (1 - alpha_sigmoid) * self.beta_gamma
                self.alpha_gamma = self.alpha_gamma.transpose(2, 3)
                self.alpha_gamma.masked_fill(mask, 0)

            if self.HP.att_prior_type == 'whai_gamma' and self.training:
                # index6
                if laysa == 0:
                    samp_wei_previous = torch.exp(scores2)
                else:
                    samp_wei_previous = samp_wei_previous
                # print('shape_samp_wei_previous', samp_wei_previous.shape)
                samp_wei_previous = self.linear_prior(samp_wei_previous)
                self.prior_scores = torch.softmax((samp_wei_previous), dim=-1) * self.beta_gamma

                self.alpha_gamma = torch.softmax((self.contextual_prior_net(key)), dim=2) * self.HP.optk * self.beta_gamma
                self.alpha_gamma = self.alpha_gamma.transpose(2, 3)
                self.alpha_gamma.masked_fill(mask, 0)

                print('mean_alpha', self.alpha_gamma.mean())
                print('alpha', self.alpha_gamma)
                print('mean_prior_score', self.prior_scores.mean())
                print('prior_score', self.prior_scores)

                self.alpha_gamma = self.prior_scores  #+ self.alpha_gamma


            if self.training:
                # u_weibull = torch.rand_like(scores)

                #0.8 0.2
                #0.75 0.25
                #0.7 0.3
                u_weibull = (0.90 - 0.10) * torch.rand_like(scores) + 0.10
                # u_weibull = torch.rand_like(scores)

                if self.HP.model_type =='whai':
                    scores_mid = torch.exp(scores2) + 0.05
                    # scores_mid = torch.exp(scores2)

                    if laysa ==0:
                        hidden_pre = torch.exp(scores2)
                        hidden = torch.log(1 + torch.exp(torch.matmul(self.w_3, hidden_pre) + self.b_3) + eps) * self.beta_gamma
                        # hidden = torch.log(1 + torch.exp(torch.nn.functional.relu(torch.matmul(self.w_3, hidden_pre) + self.b_3)) + eps) * self.beta_gamma
                        # print('pass_train')
                    else:
                        hidden = torch.log(1 + torch.exp(torch.matmul(self.w_3, hidden_pre) + self.b_3) + eps) * self.beta_gamma
                        # print('no_pass_train')
                    k_weibull = (self.prior_gamma * (torch.log(1 + torch.exp(torch.matmul(self.w_1, hidden) + self.b_1) + eps))) + scores_mid

                    self.k_weibull = k_weibull

                    # k_weibull = scores_mid
                    lambda_weibull = (self.beta_gamma * (torch.log(1 + torch.exp(torch.matmul(self.w_2, hidden) + self.b_2) + eps))) + (torch.exp(scores)/torch.exp(torch.lgamma(1 + 1.0 / k_weibull).type_as(u_weibull)))

                    # lambda_weibull = ((torch.tensor(1e-5).type(torch.float32)) * (torch.log(1 + torch.exp(torch.matmul(self.w_2, hidden) + self.b_2) + eps))) + (torch.exp(scores)/torch.exp(torch.lgamma(1 + 1.0 / k_weibull).type_as(u_weibull)))

                    self.lambda_weibull = lambda_weibull

                    # lambda_weibull = (torch.exp(scores)/torch.exp(torch.lgamma(1 + 1.0 / k_weibull).type_as(u_weibull)))
                    # lambda_weibull = (self.beta_gamma * (torch.log(1 + torch.exp(torch.matmul(self.w_2, hidden) + self.b_2) + eps))) + torch.exp(scores)

                if self.HP.model_type == 'bam_contexk':
                    k_weibull = (torch.exp(scores2) + 0.05)
                    # print('k_weibull_sum', (k_weibull < 0).sum())
                    k_weibull = k_weibull + att_map_last
                    # print('att_map_last', att_map_last)
                    # print('mean_att_map_last', att_map_last.mean())

                    lambda_weibull = torch.exp(scores) / torch.exp(torch.lgamma(1 + 1.0 / k_weibull).type_as(u_weibull))

                sample_weibull = lambda_weibull * (-torch.log(1.0 - u_weibull + eps)) ** (1.0 / k_weibull)

                att_map = sample_weibull / (sample_weibull.sum(-1, keepdim=True) + eps)
                # print('att_map', att_map)
                # print('att_map', (att_map < 0).sum())
                # print('att_map_last', (att_map_last < 0).sum())

                if self.HP.att_kl != 0.0:
                    KL = self.alpha_gamma * scores - np.euler_gamma * self.alpha_gamma / k_weibull - torch.log(k_weibull + eps) - self.beta_gamma * lambda_weibull * torch.exp(torch.lgamma(1 + 1.0 /k_weibull)) + np.euler_gamma \
                         + 1.0 + self.alpha_gamma * torch.log(torch.tensor(self.beta_gamma) + eps) - torch.lgamma(torch.tensor(self.alpha_gamma) + eps)
                    self.KL_backward = -KL.mean()
            else:
                if self.HP.model_type == 'bam_contexk':
                    att_map = F.softmax(scores, dim=-1)
                    k_weibull = att_map

                if self.HP.model_type == 'whai':
                    u_weibull = (0.90 - 0.10) * torch.rand_like(scores) + 0.10
                    # u_weibull = torch.rand_like(scores)
                    eps = 1e-20
                    scores_mid = torch.exp(scores2) + 0.05
                    # scores_mid = torch.exp(scores2)
                    if laysa == 0:
                        hidden_pre = torch.exp(scores2)
                        hidden = torch.log(1 + torch.exp(torch.matmul(self.w_3, hidden_pre) + self.b_3) + eps) * self.beta_gamma
                        # print('pass_test')
                    else:
                        hidden = torch.log(1 + torch.exp(torch.matmul(self.w_3, hidden_pre) + self.b_3) + eps) * self.beta_gamma
                        # print('no_pass_test')
                    k_weibull = (self.prior_gamma * (torch.log(1 + torch.exp(torch.matmul(self.w_1, hidden) + self.b_1) + eps))) + scores_mid
                    self.k_weibull =k_weibull
                    # lambda_weibull = ((torch.tensor(1e-5).type(torch.float32)) * (torch.log(1 + torch.exp(torch.matmul(self.w_2, hidden) + self.b_2) + eps))) + (torch.exp(scores)/torch.exp(torch.lgamma(1 + 1.0 / k_weibull).type_as(u_weibull)))

                    # k_weibull = scores_mid
                    lambda_weibull = (self.beta_gamma * (torch.log(1 + torch.exp(torch.matmul(self.w_2, hidden) + self.b_2) + eps))) + (
                                                 torch.exp(scores) / torch.exp(
                                             torch.lgamma(1 + 1.0 / k_weibull).type_as(u_weibull)))


                    self.lambda_weibull = lambda_weibull

                    att_map = lambda_weibull * (torch.exp(torch.lgamma(1 + 1.0 / k_weibull)))
                    att_map = att_map / (att_map.sum(-1, keepdim=True) + eps)
                    hidden = att_map
                    sample_weibull = att_map

                # att_map = F.softmax(scores, dim=-1)

        if self.HP.att_type == 'soft_lognormal':
            if self.HP.att_prior_type == 'contextual' and self.training:
                self.mean_normal_prior = torch.softmax((self.contextual_prior_net(key)),dim=2) * self.HP.optk #* self.beta_gamma
                self.mean_normal_prior = self.mean_normal_prior.transpose(2, 3)

            if self.training:
                eps = 1e-20
                normal_distribution = torch.distributions.normal.Normal(0, 1)
                att_map_ori = F.log_softmax(scores, dim=-1)
                self.att_map_ori = scores.data
                scores = att_map_ori
                sample_normal = scores + self.sigma_normal_posterior * normal_distribution.sample(scores.size()).cuda().type_as(scores) - self.sigma_normal_posterior ** 2 / 2
                att_map = F.softmax(sample_normal, dim=-1)
                if self.HP.att_kl != 0.0:
                    KL = torch.log(self.sigma_normal_prior / self.sigma_normal_posterior + eps) + (self.sigma_normal_posterior ** 2 + (scores - self.mean_normal_prior) ** 2) / (2 * self.sigma_normal_prior ** 2) - 0.5
                    self.KL_backward = KL.mean()
            else:
                att_map = F.softmax(scores, dim=-1)

        if self.HP.model_type == 'bam_contexk':
            return torch.matmul(att_map, value), k_weibull
        if self.HP.model_type == 'whai':
            return torch.matmul(att_map, value), hidden, sample_weibull


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, HP):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=HP.HIDDEN_SIZE,
            mid_size=HP.FF_SIZE,
            out_size=HP.HIDDEN_SIZE,
            dropout_r=HP.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, HP):
        super(SA, self).__init__()

        self.HP = HP
        in_features = 14
        out_features =14
        prior_featuresSA=14
        self.mhatt = MHAtt(HP, in_features, out_features, prior_featuresSA)
        self.ffn = FFN(HP)

        self.dropout1 = nn.Dropout(HP.DROPOUT_R)
        self.norm1 = LayerNorm(HP.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(HP.DROPOUT_R)
        self.norm2 = LayerNorm(HP.HIDDEN_SIZE)

    def forward(self, x, x_mask, att_map_last_SA, hidden, samp_wei, laysa):
        if self.HP.model_type =='bam_contexk':
            for_x, att_map_last = self.mhatt(x, x, x, x_mask, att_map_last_SA, hidden, samp_wei)
            x = self.norm1(x + self.dropout1(
                # self.mhatt(x, x, x, x_mask)
                for_x
            ))

            x = self.norm2(x + self.dropout2(
                self.ffn(x)
            ))
            return x, att_map_last

        if self.HP.model_type == 'whai':
            for_x, hidden, samp_wei = self.mhatt(x, x, x, x_mask, att_map_last_SA, hidden, samp_wei, laysa)
            x = self.norm1(x + self.dropout1(
                # self.mhatt(x, x, x, x_mask)
                for_x
            ))

            x = self.norm2(x + self.dropout2(
                self.ffn(x)
            ))

            return x, hidden, samp_wei

# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, HP):
        super(SGA, self).__init__()
        self.HP = HP
        in_features = 100
        out_features =100
        prior_features=100
        self.mhatt1 = MHAtt(HP, in_features, out_features, prior_features)
        prior_features= 14
        self.mhatt2 = MHAtt(HP, in_features, out_features, prior_features)
        self.ffn = FFN(HP)

        self.dropout1 = nn.Dropout(HP.DROPOUT_R)
        self.norm1 = LayerNorm(HP.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(HP.DROPOUT_R)
        self.norm2 = LayerNorm(HP.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(HP.DROPOUT_R)
        self.norm3 = LayerNorm(HP.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, att_map_last_SGA_1, att_map_last_SGA_2, hidden_1, hidden_2, samp_wei_1, samp_wei_2, laysga):
        if self.HP.model_type =='bam_contexk':
            for_x1, att_map_last_SGA_final_1 = self.mhatt1(x, x, x, x_mask, att_map_last_SGA_1, hidden_1, samp_wei_1)
            # print('att_map_last_1', att_map_last.shape)
            # print('att_map_last_2', self.mhatt2(y, y, x, y_mask, att_map_last_SGA_2)[1].shape)


            x = self.norm1(x + self.dropout1(
                # self.mhatt1(x, x, x, x_mask)
                for_x1
            ))

            for_x2, att_map_last_SGA_final_2 = self.mhatt2(y, y, x, y_mask, att_map_last_SGA_2, hidden_2, samp_wei_2)
            # att_map_last = torch.cat([att_map_last, att_map_mid], dim=3)
            # print('att_map_last_3', att_map_last.shape)

            x = self.norm2(x + self.dropout2(
                # self.mhatt2(y, y, x, y_mask)
                for_x2
            ))

            x = self.norm3(x + self.dropout3(
                self.ffn(x)
            ))
            return x, att_map_last_SGA_final_1, att_map_last_SGA_final_2

        if self.HP.model_type =='whai':
            for_x1, hidden_1, samp_wei_1 = self.mhatt1(x, x, x, x_mask, att_map_last_SGA_1, hidden_1, samp_wei_1, laysga)

            x = self.norm1(x + self.dropout1(
                # self.mhatt1(x, x, x, x_mask)
                for_x1
            ))

            for_x2, hidden_2, samp_wei_2 =self.mhatt2(y, y, x, y_mask, att_map_last_SGA_2, hidden_2, samp_wei_2, laysga)

            x = self.norm2(x + self.dropout2(
                # self.mhatt2(y, y, x, y_mask)
                for_x2
            ))

            x = self.norm3(x + self.dropout3(
                self.ffn(x)
            ))

            return x, hidden_1, hidden_2, samp_wei_1, samp_wei_2


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, HP):
        super(MCA_ED, self).__init__()
        self.HP = HP
        self.enc_list = nn.ModuleList([SA(HP) for _ in range(HP.LAYER)])
        self.dec_list = nn.ModuleList([SGA(HP) for _ in range(HP.LAYER)])

    def forward(self, x, y, x_mask, y_mask, att_map_last_SA, att_map_last_SGA_1, att_map_last_SGA_2, h_SA, h_SGA_1, h_SGA_2, samp_wei_SA, samp_wei_SGA_1, samp_wei_SGA_2):
        if self.HP.model_type == 'bam_contexk':
            att_map_last = att_map_last_SA
            for enc in self.enc_list:
                x, att_map_mid_SA = enc(x, x_mask, att_map_last, h_SA, samp_wei_SA)
                att_map_last = att_map_mid_SA

            att_map_last_SGA_final_1 = att_map_last_SGA_1
            att_map_last_SGA_final_2 = att_map_last_SGA_2
            for dec in self.dec_list:
                y, att_map_mid_SGA_1, att_map_mid_SGA_2 = dec(y, x, y_mask, x_mask, att_map_last_SGA_final_1,
                                                              att_map_last_SGA_final_2, h_SGA_1, h_SGA_2, samp_wei_SGA_1, samp_wei_SGA_2)
                att_map_last_SGA_final_1 = att_map_mid_SGA_1
                att_map_last_SGA_final_2 = att_map_mid_SGA_2
            return x, y

        if self.HP.model_type == 'whai':
            # hidden_last = h_SA
            # for enc in self.enc_list:
            #     x, att_map_mid_SA = enc(x, x_mask, att_map_last, h_SA)
            #     att_map_last = att_map_mid_SA

            hidden = h_SA
            sample_weibull =samp_wei_SA
            att_map_last = att_map_last_SA
            laysa=0
            for enc in self.enc_list:
                # print('enc', enc)
                # print('count_SA', laysa)
                if laysa==0:
                    x, hidden_mid_SA, sample_mid_SA = enc(x, x_mask, att_map_last, hidden, sample_weibull, laysa)
                else:
                    x, hidden_mid_SA, sample_mid_SA = enc(x, x_mask, att_map_last, hidden, sample_weibull, laysa)
                hidden = hidden_mid_SA
                sample_weibull = sample_mid_SA
                laysa=laysa+1


            att_map_last_SGA_final_1 = att_map_last_SGA_1
            att_map_last_SGA_final_2 = att_map_last_SGA_2
            hidden_SGA_1 = h_SGA_1
            hidden_SGA_2 = h_SGA_2
            sample_weibull_SGA_1 = samp_wei_SGA_1
            sample_weibull_SGA_2 = samp_wei_SGA_2
            laysga = 0
            for dec in self.dec_list:
                # print('dec', dec)
                # print('count_SGGGGGA', laysga)
                if laysga==0:
                    y, hidden_mid_SGA_1, hidden_mid_SGA_2, sample_mid_SGA_1, sample_mid_SGA_2  = dec(y, x, y_mask, x_mask, att_map_last_SGA_final_1,
                                                                  att_map_last_SGA_final_2, hidden_SGA_1, hidden_SGA_2, sample_weibull_SGA_1, sample_weibull_SGA_2, laysga)
                else:
                    y, hidden_mid_SGA_1, hidden_mid_SGA_2, sample_mid_SGA_1, sample_mid_SGA_2  = dec(y, x, y_mask, x_mask, att_map_last_SGA_final_1,
                                                                  att_map_last_SGA_final_2, hidden_SGA_1, hidden_SGA_2, sample_weibull_SGA_1, sample_weibull_SGA_2, laysga)
                hidden_SGA_1 = hidden_mid_SGA_1
                hidden_SGA_2 = hidden_mid_SGA_2
                sample_weibull_SGA_1 = sample_mid_SGA_1
                sample_weibull_SGA_2 = sample_mid_SGA_2
                laysga=laysga+1

            return x, y
