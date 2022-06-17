# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.load_data import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import scipy.stats as sts
from six.moves import cPickle


def setup_seed_2(seed):
    np.random.seed(seed)

class Execution:
    def __init__(self, HP):
        self.HP = HP

        print('Loading training set ........')
        self.dataset = DataSet(HP)

        self.dataset_eval = None
        if HP.EVAL_EVERY_EPOCH:
            HP_eval = copy.deepcopy(HP)
            setattr(HP_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(HP_eval)


    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        setup_seed_2(1)
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb
        self.HP.data_size = data_size
        print(data_size)

        # Define the MCAN model
        net = Net(
            self.HP,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        self.log_prob_list = []
        self.log_prob_list = self.log_prob_list + [enc.mhatt for enc in net.backbone.enc_list]
        self.log_prob_list = self.log_prob_list + ([dec.mhatt1 for dec in net.backbone.dec_list])
        self.log_prob_list = self.log_prob_list + ([dec.mhatt2 for dec in net.backbone.dec_list])



        # Define the multi-gpu training if needed
        if self.HP.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.HP.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()
        loss_fn_keep = torch.nn.BCELoss(reduction='none').cuda()

        # Load checkpoint if resume training
        if self.HP.RESUME:
            print(' ========== Resume training')

            if self.HP.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.HP.CKPT_PATH
            else:
                path = self.HP.CKPTS_PATH + \
                       'ckpt_' + self.HP.CKPT_VERSION + \
                       '/epoch' + str(self.HP.CKPT_EPOCH) + '.pkl'
            if self.HP.FINE_TUNE and not (os.path.exists(self.HP.CKPTS_PATH + 'ckpt_' + self.HP.VERSION)):
            # if self.HP.FINE_TUNE:
                os.mkdir(self.HP.CKPTS_PATH + 'ckpt_' + self.HP.VERSION)

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.HP, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.HP.BATCH_SIZE * self.HP.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.HP.CKPT_EPOCH

        else:
            if ('ckpt_' + self.HP.VERSION) in os.listdir(self.HP.CKPTS_PATH):
                shutil.rmtree(self.HP.CKPTS_PATH + 'ckpt_' + self.HP.VERSION)

            os.mkdir(self.HP.CKPTS_PATH + 'ckpt_' + self.HP.VERSION)

            optim = get_optim(self.HP, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.HP.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.HP.BATCH_SIZE,
                shuffle=False,
                num_workers=self.HP.NUM_WORKERS,
                pin_memory=self.HP.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.HP.BATCH_SIZE,
                shuffle=True,
                num_workers=self.HP.NUM_WORKERS,
                pin_memory=self.HP.PIN_MEM,
                drop_last=True
            )

        # Training script
        #Newnew
        # histories = {}
        # att_weight_list_epoch = histories.get('att_weight_history', {})


        for epoch in range(start_epoch, self.HP.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.HP.LOG_PATH +
                'log_run_' + self.HP.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.HP.LR_DECAY_LIST:
                adjust_lr(optim, self.HP.LR_DECAY_R)

            # Externally shuffle
            if self.HP.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()

            # att_weight_list_epoch = []
            # prior_weight_list_epoch = []

            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):

                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                #the below two line is for small training data sets
                # if step > 0:
                #     break
                for accu_step in range(self.HP.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.HP.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.HP.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.HP.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.HP.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.HP.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.HP.SUB_BATCH_SIZE]

                    if self.HP.add_noise:
                        # Broadcast version
                        gaussian_noise = np.random.normal(size=img_feat_iter.size()[1:]) * self.HP.noise_scalar
                        gaussian_noise = torch.from_numpy(gaussian_noise).type_as(img_feat_iter).cuda()
                        img_feat_iter = img_feat_iter + gaussian_noise.unsqueeze(0)

                        #Nonbroadcast version
                        # gaussian_noise = np.random.normal(size=sub_img_feat_iter.size()) * self.HP.noise_scalar
                        # gaussian_noise = torch.from_numpy(gaussian_noise).type_as(sub_img_feat_iter).cuda()
                        # sub_img_feat_iter = sub_img_feat_iter + gaussian_noise


                    # pred = net(
                    #     sub_img_feat_iter,
                    #     sub_ques_ix_iter
                    # )
                    #
                    # loss = loss_fn(pred, sub_ans_iter)
                    if self.HP.att_type == 'hard_attention':
                        if self.HP.att_optim_type == 'reinforce':
                            pred = net(
                                sub_img_feat_iter,
                                sub_ques_ix_iter
                            )
                            loss = loss_fn(pred, sub_ans_iter)
                            loss_hard = loss_fn_keep(pred, sub_ans_iter).data
                            loss_hard = loss_hard.sum(dim=1)
                            # print('loss_hard', loss_hard.shape)
                            # TODO should use loss_fn_keep.
                            loss_re_keep=0
                            for layer in self.log_prob_list:
                                layer.log_probs = layer.log_probs.sum(dim=(1, 2, 3))
                                # print('log_prob3', layer.log_probs.shape)
                                loss_re = loss_hard.mul(layer.log_probs)
                                loss_re = loss_re.sum()
                                loss_re_keep = loss_re_keep + loss_re
                                # TODO: why take mean?
                                # loss_re.backward(retain_graph=True)
                            loss = loss + loss_re_keep
                            # print('loss2', loss.shape)
                        else:
                            pred = net(
                                sub_img_feat_iter,
                                sub_ques_ix_iter
                            )
                            loss = loss_fn(pred, sub_ans_iter)
                        # TODO: minimize the definition of flag.

                    if self.HP.att_type == 'hard_var_att':
                        if self.HP.att_optim_type == 'gumbel':
                            pred = net(
                                sub_img_feat_iter,
                                sub_ques_ix_iter
                            )
                            loss_hard = loss_fn(pred, sub_ans_iter)
                            att_penalty = 0
                            # att_prior_sum = 0
                            for layer in self.log_prob_list:
                                att_nll_shape = len(layer.att_post_nll_true.shape)
                                # print('att_post_nll_true', layer.att_post_nll_true.shape)
                                att_penalty = att_penalty + layer.att_post_nll_true.mean(
                                    tuple(np.arange(1, att_nll_shape))) - layer.att_prior_nll_true.mean(
                                    tuple(np.arange(1, att_nll_shape)))
                                # TODO: why not mean for prior?
                                # TODO: att_penalty is wrong.
                                # att_prior_sum = att_prior_sum + layer.att_prior_nll_true
                            loss = loss_hard - att_penalty
                            loss = torch.sum(loss)
                        else:
                            if self.HP.att_optim_type == 'reinforce_base':
                                self.forward_mode_att(True)
                                pred = net(
                                    sub_img_feat_iter,
                                    sub_ques_ix_iter
                                )
                                loss_orig1 = loss_fn(pred, sub_ans_iter)
                                loss_hard = loss_fn_keep(pred, sub_ans_iter).data
                                loss_hard = loss_hard.sum(dim=1)
                                att_penalty = 0
                                # att_prior_sum = 0
                                # TODO: remove.
                                log_probs_all_layer = 0

                                for layer in self.log_prob_list:
                                    att_nll_shape = len(layer.att_post_nll_true.shape)
                                    att_penalty = att_penalty + layer.att_post_nll_true.mean(
                                        tuple(np.arange(1, att_nll_shape))).data - layer.att_prior_nll_true.mean(
                                        tuple(np.arange(1, att_nll_shape))).data
                                    log_probs_shape = len(layer.log_probs.shape)
                                    log_probs_all_layer = log_probs_all_layer + layer.log_probs.mean(
                                        tuple(np.arange(1, log_probs_shape)))
                                loss = loss_hard #- att_penalty  # TODO: stop gradient?
                                # print('losshard', loss_hard)
                                # print('att_penalty', att_penalty)
                                loss = loss.mul(log_probs_all_layer)
                                loss_1 = torch.sum(loss)
                                # loss_orig1.backward(retain_graph=True)

                                self.forward_mode_att(False)
                                pred = net(
                                    sub_img_feat_iter,
                                    sub_ques_ix_iter
                                )
                                att_penalty = 0
                                loss_orig2 = loss_fn(pred, sub_ans_iter)
                                loss_hard = loss_fn_keep(pred, sub_ans_iter).data
                                loss_hard = loss_hard.sum(dim=1)
                                # for layer in self.log_prob_list:
                                #     att_nll_shape = len(layer.att_post_nll_true.shape)
                                #     # print('att_post_nll_true', layer.att_post_nll_true.shape)
                                #     att_penalty = att_penalty + layer.att_post_nll_true.mean(
                                #         tuple(np.arange(1, att_nll_shape))).data - \
                                #                   layer.att_prior_nll_true.mean(tuple(np.arange(1, att_nll_shape))).data
                                loss = loss_hard - att_penalty
                                loss = loss.mul(log_probs_all_layer)
                                loss_2 = torch.sum(loss)
                                # loss_orig2.backward(retain_graph=True)
                                loss = loss_1 - loss_2 + loss_orig1 #+ loss_orig2  # TODO: you have changed the definition of loss, which would affect the following optimization.
                    # else:
                    #     pred = net(
                    #         sub_img_feat_iter,
                    #         sub_ques_ix_iter
                    #     )
                    #     loss = loss_fn(pred, sub_ans_iter)
                        if self.HP.ARM and self.HP.dp_type:
                            loss_keep = loss_fn_keep(pred, sub_ans_iter).sum(1)
                            penalty = 0
                            prior_sum = 0
                            for layer in self.dropout_list:
                                nll_shape = len(layer.post_nll_true.shape)
                                penalty = penalty + layer.post_nll_true.mean(tuple(np.arange(1, nll_shape))).data - \
                                          layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape))).data
                                prior_sum = prior_sum + layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape)))
                            if self.HP.learn_prior:
                                prior_sum.mean().backward(retain_graph=True)
                            f2 = loss_keep.data - penalty
                            self.forward_mode(False)
                            pred = net(
                                sub_img_feat_iter,
                                sub_ques_ix_iter
                            )
                            loss_keep = loss_fn_keep(pred, sub_ans_iter).sum(1)
                            penalty = 0
                            for layer in self.dropout_list:
                                nll_shape = len(layer.post_nll_true.shape)
                                penalty = penalty + layer.post_nll_true.mean(tuple(np.arange(1, nll_shape))).data - \
                                          layer.prior_nll_true.mean(tuple(np.arange(1, nll_shape))).data
                            f1 = loss_keep.data - penalty  # .data
                            self.update_phi_gradient(f1, f2)

                    if self.HP.att_type == 'soft_attention':
                        pred = net(
                            sub_img_feat_iter,
                            sub_ques_ix_iter
                        )
                        # print('pred', pred.shape)
                        # print('ans', sub_ans_iter.shape)
                        loss = loss_fn(pred, sub_ans_iter)


                    if self.HP.att_type == 'soft_weibull' or self.HP.att_type == 'soft_lognormal' or self.HP.att_type == 'gamma_att':
                        pred = net(
                            sub_img_feat_iter,
                            sub_ques_ix_iter
                        )
                        loss = loss_fn(pred, sub_ans_iter)
                        # print('sub_ans', sub_ans_iter)
                        # if self.HP.att_kl != 0.0:
                        #     att_penalty = 0
                        #     # att_prior_sum = 0
                        #     for layer in self.log_prob_list:
                        #         # att_nll_shape = len(layer.att_post_nll_true.shape)
                        #         # print('att_post_nll_true', layer.att_post_nll_true.shape)
                        #         att_penalty = att_penalty + layer.KL_backward
                        #             # tuple(np.arange(1, att_nll_shape))) - layer.att_prior_nll_true.mean(
                        #             # tuple(np.arange(1, att_nll_shape))).detach()
                        #         # TODO: why not mean for prior?
                        #         # TODO: att_penalty is wrong.
                        #         # att_prior_sum = att_prior_sum + layer.att_prior_nll_true
                        att_penalty = 0
                        if epoch >= self.HP.kl_start_epoch:
                            if self.HP.att_kl != 0.0:
                                # att_penalty = 0
                                # att_prior_sum = 0
                                for layer in self.log_prob_list:
                                    # att_nll_shape = len(layer.att_post_nll_true.shape)
                                    # print('att_post_nll_true', layer.att_post_nll_true.shape)
                                    att_penalty = att_penalty + layer.KL_backward
                                    # tuple(np.arange(1, att_nll_shape))) - layer.att_prior_nll_true.mean(
                                    # tuple(np.arange(1, att_nll_shape))).detach()
                                    # TODO: why not mean for prior?
                                    # TODO: att_penalty is wrong.
                                    # att_prior_sum = att_prior_sum + layer.att_prior_nll_true
                                epoch_diff = epoch - self.HP.kl_start_epoch
                                epoch_diff = epoch_diff * self.HP.kl_anneal_rate
                                # loss = loss + self.HP.att_kl * np.exp(epoch_diff) / (1 + np.exp(epoch_diff)) * kl / len(self.model.attention_kl_list)
                                # print('penalty1', att_penalty)
                                att_penalty = self.HP.att_kl * np.exp(epoch_diff) / (1 + np.exp(epoch_diff)) * att_penalty
                                # print('penalty2', att_penalty)

                        loss = loss + att_penalty


                    #Newnew
                    # attention_weight_list =[]
                    # prior_weight_list =[]
                    # if step < 5 and epoch == 12:
                    # # if step < 100:
                    #     for layer in self.log_prob_list:
                    #         attention_weight_list.append(layer.att_map_ori.data.cpu())
                    #         if self.HP.att_type == 'soft_weibull':
                    #             prior_weight_list.append(layer.alpha_gamma.data.cpu())
                    #         # print('list', len(attention_weight_list))
                    #
                    #     # att_weight_list_epoch['att_weight_history'].append(attention_weight_list)
                    #     att_weight_list_epoch.append(attention_weight_list)
                    #     if self.HP.att_type == 'soft_weibull':
                    #         prior_weight_list_epoch.append(prior_weight_list)
                    # print('len epcoh',len(att_weight_list_epoch))
                    # print('first',att_weight_list_epoch[0])




                    loss /= self.HP.GRAD_ACCU_STEPS
                    loss.backward(retain_graph=True)
                    loss_sum += loss.cpu().data.numpy() * self.HP.GRAD_ACCU_STEPS

                    if self.HP.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.HP.SPLIT['train'] + '->' + self.HP.SPLIT['val']
                        else:
                            mode_str = self.HP.SPLIT['train'] + '->' + self.HP.SPLIT['test']

                        if step %1000 ==0:
                            print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                                self.HP.VERSION,
                                epoch + 1,
                                step,
                                int(data_size / self.HP.BATCH_SIZE),
                                mode_str,
                                loss.cpu().data.numpy() / self.HP.SUB_BATCH_SIZE,
                                optim._rate
                            ), end='          ')

                # Gradient norm clipping
                if self.HP.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.HP.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.HP.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.HP.CKPTS_PATH +
                'ckpt_' + self.HP.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            #Newnew

            # histories['att_weight_history'] = att_weight_list_epoch
            # if self.HP.att_type == 'soft_weibull':
            #     histories['prior_weight_history'] = prior_weight_list_epoch
            # #
            #
            # if epoch_finish ==13:
            #     directory = self.HP.CKPTS_PATH +  'ckpt_' + self.HP.VERSION #+ '/epoch' + str(epoch_finish)
            #     with open(os.path.join(directory, 'histories_' + '.pkl'), 'wb') as f:
            #         cPickle.dump(histories, f)

            # Logging
            logfile = open(
                self.HP.LOG_PATH +
                'log_run_' + self.HP.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            # if dataset_eval is not None and (epoch+1) == 2:
            #     self.eval(
            #         dataset_eval,
            #         state_dict=net.state_dict(),
            #         valid=True
            #     )

            if dataset_eval is not None and (epoch+1)==13:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            #Fine_tune version
            # if dataset_eval is not None and (epoch + 1) == 16:
            #     self.eval(
            #         dataset_eval,
            #         state_dict=net.state_dict(),
            #         valid=True
            #     )

            # if self.HP.VERBOSE:
            #     logfile = open(
            #         self.HP.LOG_PATH +
            #         'log_run_' + self.HP.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.HP.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):
        setup_seed_2(1)
        elbo_list =[]
        # Load parameters
        if self.HP.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.HP.CKPT_PATH
        else:
            path = self.HP.CKPTS_PATH + \
                   'ckpt_' + self.HP.CKPT_VERSION + \
                   '/epoch' + str(self.HP.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []
        p_value_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(
            self.HP,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        self.log_prob_list = []
        self.log_prob_list = self.log_prob_list + [enc.mhatt for enc in net.backbone.enc_list]
        self.log_prob_list = self.log_prob_list + ([dec.mhatt1 for dec in net.backbone.dec_list])
        self.log_prob_list = self.log_prob_list + ([dec.mhatt2 for dec in net.backbone.dec_list])


        if self.HP.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.HP.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.HP.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.HP.NUM_WORKERS,
            pin_memory=True
        )

        histories = {}
        att_weight_list_epoch = []
        prior_weight_list_epoch = []
        k_weight_list_epoch =[]
        lambda_weight_list_epoch = []

        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.HP.EVAL_BATCH_SIZE),
            ), end='          ')

            # if step > 2:
            #     break

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            if self.HP.add_noise:
                # Broadcast version
                gaussian_noise = np.random.normal(size=img_feat_iter.size()[1:]) * self.HP.noise_scalar
                gaussian_noise = torch.from_numpy(gaussian_noise).type_as(img_feat_iter).cuda()
                img_feat_iter = img_feat_iter + gaussian_noise.unsqueeze(0)

                # nonbroadcast
                # gaussian_noise = np.random.normal(size=img_feat_iter.size()) * self.HP.noise_scalar
                # gaussian_noise = torch.from_numpy(gaussian_noise).type_as(img_feat_iter).cuda()
                # img_feat_iter = img_feat_iter + gaussian_noise




            pred = net(
                img_feat_iter,
                ques_ix_iter
            )
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            net.train()
            pred_uncertain = torch.zeros([0]).cuda()
            # attention_weight_list = []
            # pred = net(img_feat_iter, ques_ix_iter).data
            for iii in range(self.HP.uncertainty_sample):
                pred = net(img_feat_iter, ques_ix_iter).data
                loss = loss_fn(pred, ans_iter.cuda())
                att_penalty =0
                if self.HP.att_type == 'hard_var_att':
                    for layer in self.log_prob_list:
                        att_nll_shape = len(layer.att_post_nll_true.shape)
                        # print('att_post_nll_true', layer.att_post_nll_true.shape)
                        att_penalty = att_penalty + layer.att_post_nll_true.mean(
                            tuple(np.arange(1, att_nll_shape))).data - layer.att_prior_nll_true.mean(
                            tuple(np.arange(1, att_nll_shape))).data
                    elbo_list.append((-loss.cpu().data + att_penalty.cpu()).mean())

                else:
                    if (self.HP.att_type == 'soft_weibull' or self.HP.att_type == 'soft_lognormal' or self.HP.att_type == 'gamma_att') and self.HP.att_kl != 0.0:
                        for layer in self.log_prob_list:
                            att_penalty = att_penalty + layer.KL_backward.data
                        elbo_list.append((-loss.cpu().data - att_penalty.cpu()).mean())
                    else:
                        elbo_list.append((-loss.cpu().data).mean())

                # for layer in self.log_prob_list:
                #     attention_weight_list.append(layer.att_map_ori.data.cpu())

                pred_uncertain = torch.cat([pred_uncertain, pred.unsqueeze(2)], 2)

                # Newnew
                attention_weight_list = []
                prior_weight_list = []
                k_weight_list= []
                lambda_weight_list = []

                if step < 1: #and epoch == 12:
                    # if step < 100:
                    for layer in self.log_prob_list:
                        attention_weight_list.append(layer.att_map_ori.data.cpu())
                        if self.HP.att_type == 'gamma_att':
                            prior_weight_list.append(layer.alpha_gamma.data.cpu())
                            k_weight_list.append(layer.k_weibull.data.cpu())
                            lambda_weight_list.append(layer.lambda_weibull.data.cpu())
                        # print('list', len(attention_weight_list))

                    # att_weight_list_epoch['att_weight_history'].append(attention_weight_list)
                    att_weight_list_epoch.append(attention_weight_list)
                    if self.HP.att_type == 'gamma_att':
                        prior_weight_list_epoch.append(prior_weight_list)
                        k_weight_list_epoch.append(k_weight_list)
                        lambda_weight_list_epoch.append(lambda_weight_list)



            net.eval()

            p_value = np.squeeze(two_sample_test_batch(pred_uncertain, self.HP.uncertainty_sample))  # sample, batch, class

            # Save the answer index
            if pred_argmax.shape[0] != self.HP.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.HP.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            if p_value.shape[0] != self.HP.EVAL_BATCH_SIZE:
                p_value = np.pad(
                    p_value,
                    (0, self.HP.EVAL_BATCH_SIZE - p_value.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)
            p_value_list.append(p_value)


            # Save the whole prediction vector
            if self.HP.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.HP.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.HP.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

        print('')
        print('ELBO*******************', np.mean(elbo_list)*100)
        ans_ix_list = np.array(ans_ix_list).reshape(-1)
        p_value_list = np.array(p_value_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix]),
            'p_value': float(p_value_list[qix])

        }for qix in range(qid_list.__len__())]





        histories['att_weight_history'] = att_weight_list_epoch
        if self.HP.att_type == 'gamma_att':
            histories['prior_weight_history'] = prior_weight_list_epoch
            histories['k_weibull'] =k_weight_list_epoch
            histories['lambda_weibull'] = lambda_weight_list_epoch

        #
        #train epoch from beginning

        if val_ckpt_flag:
            directory = self.HP.CKPTS_PATH + 'ckpt_' + self.HP.CKPT_VERSION  # + '/epoch' + str(epoch_finish)
            with open(os.path.join(directory, 'histories_' + '.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

        #test.sh
        else:
            directory = self.HP.CKPTS_PATH +  'ckpt_' + self.HP.VERSION #+ '/epoch' + str(epoch_finish)
            with open(os.path.join(directory, 'histories_' + '.pkl'), 'wb') as f:
                cPickle.dump(histories, f)



        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.HP.CACHE_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.HP.CACHE_PATH + \
                    'result_run_' + self.HP.VERSION + \
                    '.json'

        else:
            if self.HP.CKPT_PATH is not None:
                result_eval_file = \
                    self.HP.RESULT_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.HP.RESULT_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '_epoch' + str(self.HP.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.HP.TEST_SAVE_PRED:

            if self.HP.CKPT_PATH is not None:
                ensemble_file = \
                    self.HP.PRED_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.HP.PRED_PATH + \
                    'result_run_' + self.HP.CKPT_VERSION + \
                    '_epoch' + str(self.HP.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)
            #
            #
            # # att_weight_record = {
            # # }
            # torch.save(
            #     attention_weight_list,
            #     self.HP.CKPTS_PATH +
            #     'ckpt_' + self.HP.VERSION +
            #     '/epoch' + str(epoch_finish) +
            #     '.pkl'
            # )


        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.HP.QUESTION_PATH['val']
            ans_file_path = self.HP.ANSWER_PATH['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            uncertainty_result = vqaEval.evaluate(qid_list)

            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")
            print("Overall uncertainty is: %.02f, %.02f, %.02f,\n" % (vqaEval.uncertainty['overall'][0],
                                                                      vqaEval.uncertainty['overall'][1],
                                                                      vqaEval.uncertainty['overall'][2]))


            print("Per Answer Type Uncertainty is the following:")
            for ansType in vqaEval.uncertainty['perAnswerType']:
                print("%s : %.02f, %.02f, %.02f," % (ansType, vqaEval.uncertainty['perAnswerType'][ansType][0],
                                                     vqaEval.uncertainty['perAnswerType'][ansType][1],
                                                     vqaEval.uncertainty['perAnswerType'][ansType][2]))
            print("\n")

            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.CKPT_VERSION + '.txt',
                    'a+'
                )

                with open(os.path.join(self.HP.LOG_PATH +
                    'log_run_' + self.HP.CKPT_VERSION + 'uc.pkl',), 'wb') as f:
                    cPickle.dump(uncertainty_result, f)

            else:
                print('Write to log file: {}'.format(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.HP.LOG_PATH +
                    'log_run_' + self.HP.VERSION + '.txt',
                    'a+'
                )

                with open(os.path.join(self.HP.LOG_PATH +
                    'log_run_' + self.HP.VERSION + 'uc.pkl',), 'wb') as f:
                    cPickle.dump(uncertainty_result, f)

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")
            logfile.write("Overall uncertainty is: %.02f, %.02f, %.02f\n" % (vqaEval.uncertainty['overall'][0],
                                                                             vqaEval.uncertainty['overall'][1],
                                                                             vqaEval.uncertainty['overall'][2]))
            for ansType in vqaEval.uncertainty['perAnswerType']:
                logfile.write("%s : %.02f, %.02f, %.02f\n" % (ansType, vqaEval.uncertainty['perAnswerType'][ansType][0],
                                                              vqaEval.uncertainty['perAnswerType'][ansType][1],
                                                              vqaEval.uncertainty['perAnswerType'][ansType][2]))

            logfile.close()


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.HP.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def forward_mode_att(self, mode):
        for layer in self.log_prob_list:
            layer.forward_mode_att = mode


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.HP.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.HP.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')


def two_sample_test_batch(prob, sample_num):
    probmean = torch.mean(prob,2)
    values, indices = torch.topk(probmean, 2, dim=1)
    aa = prob.gather(1, indices[:,0].unsqueeze(1).unsqueeze(1).repeat(1,1,sample_num))
    bb = prob.gather(1, indices[:,1].unsqueeze(1).unsqueeze(1).repeat(1,1,sample_num))
    pvalue = sts.ttest_ind(aa.cpu(),bb.cpu(), axis=2, equal_var=False).pvalue
    return pvalue