import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import DPN
from measure import compute_measure

from thop import profile

# import EarlyStopping
from pytorchtools import EarlyStopping

import random

from tensorboardX import SummaryWriter
writer = SummaryWriter("tensorboardX/DPL-Net_1e3_train_fig")

def seed_torch(seed=42):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    # torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速
seed_torch(seed=42)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class Solver(object):
    def __init__(self, args, data_loader, data_Valloader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.data_Valloader = data_Valloader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        # self.REDCNN = RED_CNN()
        self.DPN = DPN()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.DPN = nn.DataParallel(self.DPN)
        self.DPN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()  # L1_Charbonnier_loss()
        self.optimizer = optim.Adam(self.DPN.parameters(), self.lr, betas=(0.5, 0.999))


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'DPL-Net_1e3_train_{}iter.ckpt'.format(iter_))
        torch.save(self.DPN.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'DPL-Net_1e3_train_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.DPN.load_state_dict(state_d)
        else:
            self.DPN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'DPL-Net1e3trainfig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        valloss = 1000

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for epoch in range(1, self.num_epochs):
            self.DPN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.DPN(x)
                # flops, params = profile(self.DPN, inputs=(x, ))
                # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
                # print("params=", str(params/1e6)+'{}'.format("M"))
                loss = self.criterion(pred, y)
                self.DPN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'DPL-Net_1e3_train_loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
                # compute PSNR, SSIM, RMSE
            ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
            pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    
            with torch.no_grad():
                for i, (x, y) in enumerate(self.data_Valloader):

                    shape_ = x.shape[-1]
                    x = x.unsqueeze(0).float().to(self.device)
                    y = y.unsqueeze(0).float().to(self.device)

                    if self.patch_size: # patch training
                        x = x.view(-1, 1, self.patch_size, self.patch_size)
                        y = y.view(-1, 1, self.patch_size, self.patch_size)
    
                    pred = self.DPN(x)
                    valloss = self.criterion(pred, y)

                    writer.add_scalar('val_loss', valloss, global_step=epoch)
    
                    # denormalize, truncate
                    x = self.trunc(self.denormalize_(x.cpu().detach()))
                    y = self.trunc(self.denormalize_(y.cpu().detach()))
                    pred = self.trunc(self.denormalize_(pred.cpu().detach()))
    
                    data_range = self.trunc_max - self.trunc_min
    
                    original_result, pred_result = compute_measure(x, y, pred, data_range)
                    ori_psnr_avg += original_result[0]
                    ori_ssim_avg += original_result[1]
                    ori_rmse_avg += original_result[2]
                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]
    
            # if epoch > 200:
            #     early_stopping(valloss, self.DPN)

            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break


    def test(self):
        pred_psnr_std = []
        pred_ssim_std = []
        pred_rmse_std = []
        del self.DPN
        # load
        self.DPN = DPN().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):

                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.DPN(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                pred_psnr_std.append(pred_result[0])
                pred_ssim_std.append(pred_result[1])
                pred_rmse_std.append(pred_result[2])

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader)))
            print('Predictions === \nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(np.std(pred_psnr_std), 
                                                                                                  np.std(pred_ssim_std), 
                                                                                                  np.std(pred_rmse_std)))
