import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import time

from model.HAI_models import HAIMNet_VGG
from data import test_dataset

torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './test_dataset/'

model = HAIMNet_VGG()
model.load_state_dict(torch.load('./models/HAIMNet_VGG/HAI.pth.59'))
model.cuda()
model.eval()

test_datasets = ['STERE', 'NJU2K', 'DES', 'NLPR', 'SIP']
# test_datasets = ['STERE', 'NJU2K', 'DES', 'NLPR', 'SIP', 'DUT-RGBD', 'ReDWeb']
# test_datasets = ['VT821']

for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    depth_root = dataset_path + dataset + '/depth/'
    print(dataset)
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, depth, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        time_start = time.time()
        res, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(image,depth)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))
