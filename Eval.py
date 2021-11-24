import os
import time

from tqdm import tqdm
import numpy as np
import imageio
import glob
import cv2

import torch
from torch.autograd import Variable

from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
from Utils import mor_utils


torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

visuals = 'test_outputs/'
os.makedirs(visuals, exist_ok=True)

modelSaveLoc = 'model/real_world_model.t7'

data_root = '/home/pdas/Experiments/TrimBot/test_around_garden/uvc_camera_cam_0/'
query_fmt = 'png'

batch_size = 1
nthreads = 4
if batch_size < nthreads:
    nthreads = batch_size

done = u'\u2713'

print('[I] STATUS: Create utils instances...', end='')
support = mor_utils(device)
print(done)

print('[I] STATUS: Load Network and transfer to device...', end='')
net = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(device)
net, _, _ = support.loadModels(net, modelSaveLoc)
net.to(device)
print(done)

def readFile(name):
    im = imageio.imread(name)
    rgb = im.astype(np.float32)
    rgb[np.isnan(rgb)] = 0
    rgb = cv2.resize(rgb, (256, 256))
    rgb = rgb / 255

    rgb = rgb.transpose((2, 0, 1))
    return rgb

def Eval(net):
    net.eval()

    files = glob.glob(data_root + '*.%s' % query_fmt)
    print('Found %d files at query location' % len(files))

    for data in tqdm(files):

        data = data.split('/')[-1].split('.')[0]
        img = readFile(data_root + data + '.%s' % query_fmt)
        rgb = Variable(torch.from_numpy(img).float()).to(device)
        rgb = rgb.unsqueeze(0)
        [b, c, w, h] = rgb.shape

        net_time = time.time()
        with torch.no_grad():
            pred = net(rgb)

        net_timed = time.time() - net_time

        for j in range(b):
            pred_dict = {'pred_alb': pred['reflectance'][j, :, :, :],
                         'img': rgb[j, :, :, :],
                         'pred_shd': pred['shading'][j, :, :, :],
                        }
            support.dumpOutputs3(visuals, pred_dict, filename=data, Train=False)


print('[*] Beginning Testing:')
print('\tVisuals Dumped at: ', visuals)

Eval(net)
