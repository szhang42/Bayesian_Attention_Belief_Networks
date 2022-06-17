# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.base_cfgs import Cfgs
from core.exec import Execution
import argparse, yaml


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--ARM', dest='ARM',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--learn_k_weibull', type=int, default=0,
                        help='whether to learn k in weibull distribution in variational attention.')

    parser.add_argument('--att_kl', type=float, default=0.0,
                        help='weights for KL term in variational attention.')

    parser.add_argument('--att_prior_type', type=str, default='constant',
                        help='constant, parameter, contextual, whai_gamma, which prior used in variational attention.')

    parser.add_argument('--optk', type=float, default=1.0,
                        help='k in prior sigmoid function')

    parser.add_argument('--alpha_gamma', type=float, default=1.0,
                        help='initialization of alpha in gamma distribution.')

    parser.add_argument('--beta_gamma', type=float, default=1.0,
                        help='initialization of beta in gamma distribution.')

    parser.add_argument('--prior_gamma', type=float, default=1.0,
                        help='initialization of prior in gamma distribution.')


    parser.add_argument('--kl_start_epoch', type=float, default=0.0,
                    help='epoch that starts to add kl term.')


    parser.add_argument('--kl_anneal_rate', type=float, default=1.0,
                    help='KL anneal rate.')


    parser.add_argument('--FINE_TUNE', dest='FINE_TUNE',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--UNCERTAINTY_SAMPLE', dest='uncertainty_sample',
                        default=20, type=int)


    parser.add_argument('--add_noise', dest='add_noise',
                        choices=[1, 0],
                        default=0, type=int)


    parser.add_argument('--noise_scalar', dest='noise_scalar',
                        default=0.0, type=float)


    parser.add_argument('--concrete_temp', dest='concrete_temp',
                        default=0.1, type=float)

    parser.add_argument('--att_type', dest='att_type',
                      choices=['hard_attention', 'soft_attention', 'hard_var_att', 'soft_weibull', 'soft_lognormal', 'gamma_att'],
                      help='{hard_attention, soft_attention, hard_var_att, soft_weibull, soft_lognormal, gamma_att}',
                      default='soft_attention', type=str)


    parser.add_argument('--model_type', dest='model_type',
                      choices=['bam_contexk', 'whai'],
                      help='{bam_contexk, whai}',
                      default='bam_contexk', type=str)

    parser.add_argument('--w_1', type=float, default=1.0,
                        help='weights for k.')

    parser.add_argument('--w_2', type=float, default=1.0,
                        help='weights for lambda.')

    parser.add_argument('--w_3', type=float, default=1.0,
                        help='weights for h.')


    parser.add_argument('--b_1', type=float, default=1.0,
                        help='bias for k.')

    parser.add_argument('--b_2', type=float, default=1.0,
                        help='bias for lambda.')

    parser.add_argument('--b_3', type=float, default=1.0,
                        help='bias for h.')


    parser.add_argument('--sigma_normal_prior', type=float, default=1.0,
                    help='initialization of sigma in prior normal distribution.')

    parser.add_argument('--sigma_normal_posterior', type=float, default=1.0,
                    help='initialization of sigma in posterior normal distribution.')



    parser.add_argument('--k_weibull', type=float, default=1000.0,
                    help='initialization of k in weibull distribution.')

    parser.add_argument('--lambda_weibull', type=float, default=1000.0,
                    help='initialization of lambda in weibull distribution.')


    parser.add_argument('--att_optim_type', dest='att_optim_type',
                      choices=['gumbel', 'reinforce', 'reinforce_base'],
                      help='{gumbel, reinforce, reinforce_base}',
                      default='gumbel', type=str)


    parser.add_argument('--att_contextual_se', type=int, default=0,
                        help='whether to use squeeze and excite in prior computation.')

    parser.add_argument('--att_se_hid_size', type=int, default=10,
                        help='squeeze and excite factor in attention prior.')

    parser.add_argument('--att_se_nonlinear', type=str, default='relu',
                        help='which type nonlinearity in se unit.')



    parser.add_argument('--small_validation', dest='small_validation',
                        default=0, type=int)


    parser.add_argument('--hard_gumbel', dest='hard_gumbel',
                        choices=[1, 0],
                        default=0, type=int)


    parser.add_argument('--var_gumbel', dest='var_gumbel',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--var_rein_base', dest='var_rein_base',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--Reinforce', dest='Reinforce',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=['small', 'large'],
                      help='{small, large}',
                      default='small', type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",
                      type=bool)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      help='set True to save the '
                           'prediction vectors'
                           '(only work in testing)',
                      type=bool)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size during training',
                      type=int)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                      help='max training epoch',
                      type=int)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                      help='pre-load the features into memory'
                           'to increase the I/O speed',
                      type=bool)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu select, eg.'0, 1, 2'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      help='resume training',
                      type=bool)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      help='use pin memory',
                      type=bool)

    parser.add_argument('--VERB', dest='VERBOSE',
                      help='verbose print',
                      type=bool)

    parser.add_argument('--DATA_PATH', dest='DATASET_PATH',
                      help='vqav2 dataset root path',
                      type=str)

    parser.add_argument('--FEAT_PATH', dest='FEATURE_PATH',
                      help='bottom up features root path',
                      type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    HP = Cfgs()

    args = parse_args()
    args_dict = HP.parse_to_dict(args)

    cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = {**yaml_dict, **args_dict}
    HP.add_args(args_dict)
    HP.proc()

    print('Hyper Parameters:')
    print(HP)

    HP.check_path()

    execution = Execution(HP)
    execution.run(HP.RUN_MODE)




