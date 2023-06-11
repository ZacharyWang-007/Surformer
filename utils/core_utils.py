from argparse import Namespace
import os

from sksurv.metrics import concordance_index_censored

import torch

from utils.utils import *
from torch.optim import lr_scheduler
from datasets.dataset_generic import save_splits
from models.model_set_mil import *


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    print('\nTraining Fold {}!'.format(cur))

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    reg_fn = None

    print('\nInit Model...', end=' ')

    model_dict = {'n_classes': args.n_classes}
    model = MIL_Attention_FC_surv(**model_dict).cuda()

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split,
                                    training=True,
                                    testing=args.testing,
                                    weighted=args.weighted_sample,
                                    mode=args.mode,
                                    batch_size=args.batch_size)

    val_loader = get_split_loader(val_split,
                                  testing=args.testing,
                                  mode=args.mode,
                                  batch_size=args.batch_size)

    train_loss_list, c_index_list = [], []
    for epoch in range(args.max_epochs):
        train_loss_survival = train_loop_survival(args, epoch, model, train_loader, optimizer, scheduler, loss_fn, reg_fn, args.lambda_reg, args.gc)
        c_index = validate_survival(epoch, model, val_loader, loss_fn, reg_fn, args.lambda_reg)
        if epoch >= args.max_epochs - 4:
            train_loss_list.append(train_loss_survival)
            c_index_list.append(c_index)

    index = train_loss_list.index(min(train_loss_list))
    c_index_small_loss = c_index_list[index]
    c_index_final = c_index_list[-1]

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    # model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)

    print('Val c_index_small_loss: {:.4f} c_index_final {:.4f}'.format(c_index_small_loss, c_index_final))
    return c_index_small_loss, c_index_final


def train_loop_survival(args, epoch, model, loader, optimizer, scheduler, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, label, event_time, c, path) in enumerate(loader):
        if torch.equal(data_WSI, torch.ones(1)):
            continue
        if data_WSI.size(0) < 1000:
            continue

        data_WSI = data_WSI.cuda()
        label = label.cuda()
        c = c.cuda()

        # print(data_WSI.size())

        hazards_gl, S_gl, hazards_lo, S_lo, hazards_ov, S_ov, x_local = model(x_path=data_WSI)
        S = S_ov

        if args.margin_loss:
            feats = F.normalize(x_local, dim=1)
            num_prototype = feats.size(0)
            dist = torch.matmul(feats, feats.t())
            tri_mask = torch.triu(torch.ones(num_prototype, num_prototype), diagonal=1).cuda()
            mask_margin = (dist > args.margin).int()
            mask = mask_margin * tri_mask
            cos_dist = torch.matmul(feats, feats.t()) * mask
            if mask.sum() == 0:
                cos_dist = 0.0
            else:
                cos_dist = cos_dist.sum() / mask.sum()
        else:
            feats = F.normalize(x_local, dim=1)
            num_prototype = feats.size(0)
            tri_mask = torch.triu(torch.ones(num_prototype, num_prototype), diagonal=1).cuda()
            cos_dist = torch.matmul(feats, feats.t()) * tri_mask
            cos_dist = cos_dist.sum() / tri_mask.sum()

        loss = loss_fn(hazards=hazards_gl, S=S_gl, Y=label, c=c) + loss_fn(hazards=hazards_ov, S=S_ov, Y=label, c=c) \
               + loss_fn(hazards=hazards_lo, S=S_lo, Y=label, c=c) + cos_dist

        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    scheduler.step()

    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch,
                                                                                                 train_loss_surv,
                                                                                                 train_loss,
                                                                                                 c_index))
    return train_loss_surv


def validate_survival(epoch, model, loader, loss_fn=None, reg_fn=None, lambda_reg=0.):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, label, event_time, c, path) in enumerate(loader):
        if torch.equal(data_WSI, torch.ones(1)):
            continue
        if data_WSI.size(0) < 1000:
            continue

        data_WSI = data_WSI.cuda()
        label = label.cuda()
        c = c.cuda()

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI)  # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    print('val/loss_surv, {}, {}'.format(val_loss_surv, epoch))
    print('val/c-index: {}, {}'.format(c_index, epoch))

    return c_index


def summary_survival(model, loader, n_classes):
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, label, event_time, c, _) in enumerate(loader):

        if torch.equal(data_WSI, torch.ones(1)):
            continue
        if data_WSI.size(0) < 1000:
            continue

        data_WSI = data_WSI.cuda()
        label = label.cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(x_path=data_WSI)

        risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.asscalar(event_time)
        c = np.asscalar(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]
    return patient_results, c_index
