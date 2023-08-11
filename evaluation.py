# group 11
# evaluation program

import os
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import torch



# evaluation function, fold1：input, fold2: ground truth, return value of psnr and ssim
def eval(fold1, fold2):
    eval_psnr = []
    eval_ssim = []
    eval_msssim = []

    for img_name in os.listdir(fold1):
        img1 = cv2.imread(fold1 + "/" + img_name, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(fold2 + "/" + img_name, cv2.IMREAD_UNCHANGED)
        psnr = peak_signal_noise_ratio(img1, img2)
        eval_psnr.append(psnr)
        ssim_e = structural_similarity(img1, img2, multichannel=True, win_size=3)
        eval_ssim.append(ssim_e)
    mean_psnr = sum(eval_psnr) / len(eval_psnr)
    mean_ssim = sum(eval_ssim) / len(eval_ssim)
    print("done")
    return mean_psnr, mean_ssim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(eval("MFPSNet/data/gt","MFPSNet/results_parse/MFPS_demo/iter1"))

if __name__ =="__main__":
    main()
