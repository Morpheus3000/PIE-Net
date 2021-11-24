import os
import imageio
import numpy as np
import torch
import torch.nn as nn


class mor_utils:

    def __init__(self, device):
        self.device = device

    def printTensorList(self, data):
        if isinstance(data, dict):
            print('Dictionary Containing: ')
            print('{')
            for key, tensor in data.items():
                print('\t', key, end='')
                print(' with Tensor of Size: ', tensor.size())
            print('}')
        else:
            print('List Containing: ')
            print('[')
            for tensor in data:
                print('\tTensor of Size: ', tensor.size())
            print(']')

    def saveModels(self, model, optims, iterations, path):
        if isinstance(model, nn.DataParallel):
            checkpoint = {
                'iters': iterations,
                'model': model.module.state_dict(),
                'optimizer': optims.state_dict()
            }
        else:
            checkpoint = {
                'iters': iterations,
                'model': model.state_dict(),
                'optimizer': optims.state_dict()
            }
        torch.save(checkpoint, path)

    def loadModels(self, model, path, optims=None, Test=True):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        if not Test:
            optims.load_state_dict(checkpoint['optimizer'])
        return model, optims, checkpoint['iters']

    def dumpOutputs(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            imageio.imwrite((vis + '/%s_pred_alb.png') % filename, pred_a)
            imageio.imwrite((vis + '/%s_pred_shd.png') % filename, pred_s)

    def dumpOutputs3(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            for k, ele in preds.items():
                pred = ele.cpu().detach().clone().numpy()
                pred[pred < 0] = 0
                pred = (pred / pred.max()) * 255
                pred = pred.transpose((1, 2, 0))
                pred = pred.astype(np.uint8)
                imageio.imwrite((vis + '/%s_%s.png') % (filename, k), pred)
