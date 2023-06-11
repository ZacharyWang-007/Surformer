import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import torch
import random
import numpy as np

from timeit import default_timer as timer

# Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.logger import Logger
from utils.core_utils import train
from utils.utils import get_custom_exp_code


# Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
# Pathing Parameters
parser.add_argument('--data_root_dir', type=str, default='/Workspace/zhikangwang/Datasets/TCGA/Lung', help='data directory')
parser.add_argument('--split_dir', type=str, default='tcga_luad', help='Which cancer for training.')

parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')

parser.add_argument('--results_dir', type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits', type=str, default='5foldcv')


# Model Parameters.
parser.add_argument('--model_type', type=str, choices=['deepset', 'amil', 'mi_fcn', 'dgc', 'patchgcn'], default='amil')
parser.add_argument('--mode', type=str, choices=['path', 'cluster', 'graph'], default='path')

parser.add_argument('--num_gcn_layers', type=int, default=4, help='# of GCN layers to use.')
parser.add_argument('--edge_agg', type=str, default='spatial', help="What edge relationship to use for aggregation.")
parser.add_argument('--resample', type=float, default=0.00, help='Dropping out random patches.')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')

# Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')  # 32
parser.add_argument('--max_epochs', type=int, default=40, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr_step', type=int, default=20, help='lr step')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'],
                    default='nll_surv', help='slide-level classification loss function (default: ce)')

parser.add_argument('--margin_loss', type=bool, default=False)
parser.add_argument('--margin', type=float, default=0.2)

parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None',
                    help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
parser.add_argument('--dataset_path', default='dataset_csv', type=str)
parser.add_argument('--task_type', default='survival', type=str)

parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

args = parser.parse_args()


def main():
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    cindex_small_loss_list, cindex_final_list = [], []
    folds = np.arange(0, args.k)

    for i in folds:
        # Gets the Train + Val Dataset Loader.
        seed_torch(args.seed)
        train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'
                                                           .format(args.split_dir, i))

        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        # Run Train-Val on Survival Task.
        cindex_small_loss, cindex_final = train(datasets, i, args)
        cindex_small_loss_list.append(cindex_small_loss)
        cindex_final_list.append(cindex_final)

    print('c-index small loss {}'.format(cindex_small_loss_list))
    print('Average c-index small loss {}'.format(sum(cindex_small_loss_list)/len(cindex_small_loss_list)))
    print('/n')
    print('c-index final {}'.format(cindex_final_list))
    print('Average c-index final {}'.format(sum(cindex_final_list) / len(cindex_final_list)))


# Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # set the seed
    seed_torch(args.seed)
    files = len(os.listdir('./log')) + 1
    sys.stdout = Logger('./log/' + str(files) + '.txt')

    # Creates Experiment Code from argparse + Folder Name to Save Results
    args = get_custom_exp_code(args)
    args.task = args.split_dir + '_' + args.task_type
    print("Experiment Name:", args.exp_code)

    encoding_size = 1024
    settings = {'num_splits': args.k,
                'task': args.task,
                'max_epochs': args.max_epochs,
                'results_dir': args.results_dir,
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'bag_weight': args.bag_weight,
                'seed': args.seed,
                'model_type': args.model_type,
                'weighted_sample': args.weighted_sample,
                'gc': args.gc,
                'opt': args.opt}
    print('\nLoad Dataset')

    dataset = Generic_MIL_Survival_Dataset(
        csv_path='./%s/%s_all_clean.csv.zip' % (args.dataset_path, args.split_dir),
        mode=args.mode,
        data_dir=os.path.join(args.data_root_dir, '%s_20x_features' % args.split_dir),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=False,
        n_bins=4,
        label_col='survival_months',
        )

    # Creates results_dir Directory.
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code,
                                    str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    # Sets the absolute path of split_dir
    args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
    print("split_dir", args.split_dir)
    assert os.path.isdir(args.split_dir)
    settings.update({'split_dir': args.split_dir})

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))

    main()
    end = timer()
    print("finished!")
