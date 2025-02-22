import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from dataset.loader import Real, Syn
from model.cbdnet import Network,fixed_loss
# import EarlyStopping
from pytorchtools import EarlyStopping

from DPN import DPN

print(torch.cuda.device_count())
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--bs', default=2, type=int, help='batch size')
parser.add_argument('--ps', default=256, type=int, help='patch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', default=5000, type=int, help='sum of epochs')
args = parser.parse_args()

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=20, verbose=True)


def train(train_loader, val_loader, model, criterion, optimizer):
	losses = AverageMeter()
	losses_val = AverageMeter()
	model.train()

	for (noise_img, clean_img, sigma_img, flag) in train_loader:
		input_var = noise_img.cuda()
		target_var = clean_img.cuda()
		sigma_var = sigma_img.cuda()
		flag_var = flag.cuda()

		output = model(input_var)

		loss = criterion(output, target_var)
		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	model.eval()
	with torch.no_grad():
		for (noise_img_val, clean_img_val, sigma_img_val, flag_val) in val_loader:
			input_var_val = noise_img_val.cuda()
			target_var_val = clean_img_val.cuda()
			sigma_var_val = sigma_img_val.cuda()
			flag_var_val = flag_val.cuda()

			output_val = model(input_var_val)

			loss_val = criterion(output_val, target_var_val)
			losses_val.update(loss_val.item())

	return losses.avg,losses_val.avg


if __name__ == '__main__':
	save_dir = './save_model/'

	model = DPN()
	model.cuda()
	model = nn.DataParallel(model)

	if os.path.exists(os.path.join(save_dir, 'DPLNet_MRI15GP.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'DPLNet_MRI15GP.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'DPLNet_MRI15GP.pth.tar'))
		model.load_state_dict(model_info['state_dict'],strict=False)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		# optimizer.load_state_dict(model_info['optimizer'])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		# scheduler.load_state_dict(model_info['scheduler'])
		cur_epoch = 0
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		cur_epoch = 0
		
	criterion = nn.MSELoss()
	criterion.cuda()

	train_dataset = Real('./data/MRIGP15_train/', 4712, args.ps)
	val_dataset = Real('./data/MRIGP15_val/', 1000, args.ps)
	print("Train train_dataset")
	print(train_dataset.__len__())
	print("Train val_dataset")
	print(val_dataset.__len__())
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

	for epoch in range(cur_epoch, args.epochs + 1):
		loss,loss_val = train(train_loader, val_loader, model, criterion, optimizer)
		scheduler.step()

		# if epoch%100 == 0:
		torch.save({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler.state_dict()},
			os.path.join(save_dir, 'DPLNet_MRI15GP.pth.tar'))
		# early_stopping needs the validation loss to check if it has decresed,
		# and if it has, it will make a checkpoint of the current model
		early_stopping(loss_val, model)

		if early_stopping.early_stop:
			print("Early stopping")
			break

		print('Epoch [{0}]\t'
			'lr: {lr:.6f}\t'
			'Loss: {loss:.5f}\t'
			'valLoss:{loss_val:.5f}'
			.format(
			epoch,
			lr=optimizer.param_groups[-1]['lr'],
			loss=loss,
			loss_val=loss_val))
