import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2

from DPN import DPN
from model.cbdnet import Network
# from model.BioInhicbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw
# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from math import sqrt

print(torch.cuda.device_count())
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('input_filename', default="./data/JSRT_val/", type=str)
parser.add_argument('output_filename', default="./data/JSRT_result/", type=str)
args = parser.parse_args()


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, data_range=None):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        psnr = psnr.cpu().numpy()
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    while len(tensor.size()) < 4:
        tensor = tensor.unsqueeze(1)
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    # output_tf = np.array(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += compare_ssim(target_tf[im_idx, ...], output_tf[im_idx, ...], multichannel=True, data_range=255)
        n += 1.0
    return ssim / n

save_dir = './save_model/'

model = DPN()
model.cuda()
model = nn.DataParallel(model)
torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model.eval()

if os.path.exists(os.path.join(save_dir, 'MIPNJSRT2e4_batch1_patchsize256.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'MIPNJSRT2e4_batch1_patchsize256.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

psnr_val_rgb = []
ssim_val_rgb = []
rmse_val_rgb = []
for img in os.listdir(args.input_filename+'/GT_SRGB/'):
    # print(args.input_filename+'/clean/'+img)
    input_image = read_img(args.input_filename+'/GT_SRGB/'+img)
    # noise = np.random.normal(0, 50. / 255, input_image.shape)
    # print(args.input_filename+'/noisy/'+img)
    noise_image = read_img(args.input_filename+'/NOISY_SRGB/'+img)
    input_var = torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
    noise_var = torch.from_numpy(hwc_to_chw(noise_image)).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(noise_var)
    psnr_val_rgb.append(batch_PSNR(output, input_var, 1.))
    ssim_val_rgb.append(calculate_ssim(output, input_var))

    output = torch.clamp(output,0,1)
    input_var = torch.clamp(input_var,0,1)
    output = output.cpu().numpy()
    input_var = input_var.cpu().numpy()
    output = output.astype('float64')
    input_var = input_var.astype('float64')
    mse = np.sum((input_var - output) ** 2) / len(input_var)
    rmse = sqrt(mse)
    rmse_val_rgb.append(rmse)

    output_image = chw_to_hwc(output[0,...])
    output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

    cv2.imwrite(args.output_filename+img, output_image)

psnr_val = sum(psnr_val_rgb)/len(psnr_val_rgb)
psnr_std = np.std(psnr_val_rgb)
ssim_val = sum(ssim_val_rgb)/len(ssim_val_rgb)
ssim_std = np.std(ssim_val_rgb)
rmse_val = sum(rmse_val_rgb)/len(rmse_val_rgb)
rmse_std = np.std(rmse_val_rgb)
print("PSNR: %.3f " %(psnr_val))
print("PSNR Std: %.3f " %(psnr_std))
print("SSIM: %.3f " %(ssim_val))
print("SSIM Std: %.3f " %(ssim_std))
print("RMSE: %.3f " %(rmse_val))
print("RMSE Std: %.3f " %(rmse_std))