from unet_pudipflow import UNet3D_PUDIPFlow
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import tqdm
import h5py
import random
import os

scaler = GradScaler()

def Wop(X):
    return X - torch.div(X + 2 * torch.pi / 2, 2 * torch.pi, rounding_mode="floor") * torch.pi * 2

def FD3D(X, device, ttv=1):
    B, C, FE, PE, SPE = X.shape
    X_dC = X[:, 1:] - X[:, :-1]
    X_dC_bc = torch.cat([X_dC, X[:, 0:1] - X[:, -1:]], dim=1)
    X_dFE = X[:, :, 1:, :] - X[:, :, :-1, :]
    X_dFE_bc = torch.cat([X_dFE, torch.zeros((B, C, 1, PE, SPE)).to(device)], dim=2)
    X_dPE = X[:, :, :, 1:] - X[:, :, :, :-1]
    X_dPE_bc = torch.cat([X_dPE, torch.zeros((B, C, FE, 1, SPE)).to(device)], dim=3)
    X_dSPE = X[:, :, :, :, 1:] - X[:, :, :, :, :-1]
    X_dSPE_bc = torch.cat([X_dSPE, torch.zeros((B, C, FE, PE, 1)).to(device)], dim=4)
    data_fd = torch.concat([X_dC_bc.unsqueeze(1) * ttv, X_dFE_bc.unsqueeze(1), X_dPE_bc.unsqueeze(1), X_dSPE_bc.unsqueeze(1)],
                    dim=1)
    return data_fd


def NRMSE(pred, ref, mask):
    return torch.sqrt(torch.mean((pred[mask != 0] - ref[mask != 0]) ** 2)) / (
            torch.max(ref[mask != 0]) - torch.min(ref[mask != 0]))

def PUDIP(wrapped_data, weightmask, args, segmask=None, gt=None, show_SPE=10, show_T=20):

    args_dict = vars(args)
    level = args.level
    venc = args.venc
    device = args.device
    level = args.level
    ft = args.featurenum
    input_depth = args.inputdepth
    LR = args.lr
    num_iter = args.iternum
    ttv = args.TemporalTV
    vst, vsize, tst, tsize = args.vstart, args.vsize, args.tstart, args.tsize
    losstype = args.loss

    orishape = wrapped_data.shape
    Nv, Nt, FE, PE, SPE = wrapped_data.shape
    maskidx = np.where(weightmask[0, 0] == 1)
    if len(maskidx[0]) != 0:
        minx, maxx, miny, maxy, minz, maxz = np.min(maskidx[0]), np.max(maskidx[0]) + 1, np.min(maskidx[1]), np.max(
            maskidx[1]) + 1, np.min(maskidx[2]), np.max(maskidx[2]) + 1
        ans = np.zeros_like(wrapped_data)
        weightmask = weightmask[..., minx:maxx, miny:maxy, minz:maxz]
        wrapped_data = wrapped_data[..., minx:maxx, miny:maxy, minz:maxz]
        if segmask is not None:
            segmask = segmask[..., minx:maxx, miny:maxy, minz:maxz]
        if gt is not None:
            gt = gt[..., minx:maxx, miny:maxy, minz:maxz]
    else:
        minx, miny, minz = 0, 0, 0
        maxx, maxy, maxz = FE, PE, SPE
    Nv, Nt, FE, PE, SPE = wrapped_data.shape
    pads = ((0, 0), (0, 0))
    for pad in [FE, PE, SPE]:
        new = ((pad + 2 ** level - 1) // 2 ** level) * 2 ** level
        padl = (new - pad) // 2
        padr = new - pad - padl
        if pad < 2 ** level:
            pads += ((padl, padr),)
        else:
            pads += ((0, 0),)
    wrapped_data = np.pad(wrapped_data, pads, mode='constant',
                 constant_values=0)
    weightmask = np.pad(weightmask, pads, mode='constant',
                 constant_values=0)
    if segmask is not None:
        segmask = np.pad(segmask, pads, mode='constant',
                 constant_values=0)
    if gt is not None:
        gt = np.pad(gt, pads, mode='constant',
                 constant_values=0)

    show_SPE = show_SPE + pads[-1][0] - minz

    weightmask = torch.as_tensor(weightmask).to(torch.float32).to(device)
    weightmask /= torch.max(weightmask)
    B, C, FE, PE, SPE = wrapped_data.shape
    usemask = np.ones_like(np.abs(wrapped_data))
    model = UNet3D_PUDIPFlow(input_depth, tsize,ft).to(device)
    noise_like = torch.empty(B, input_depth, FE, PE, SPE).to(device)
    g_input = torch.zeros_like(noise_like).normal_() * 1e-1
    wrapped_data = torch.as_tensor(wrapped_data).to(torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data_fd = FD3D(wrapped_data, device, ttv)
    data_fd_mod = Wop(data_fd)
    loop = tqdm.tqdm(range(1, num_iter + 1), total=num_iter)
    variance = []
    ii = []
    buffer_size = 50
    model.train()
    if gt is not None and segmask is not None:
        gt = torch.from_numpy(gt).to(device)
        segmask = torch.from_numpy(segmask).to(device)
        bestn = 0
        nrmse = []
    for i in loop:
        optimizer.zero_grad()
        with autocast():
            output_unwrap = model(g_input)
            output_fd = FD3D(output_unwrap, device, ttv)
            unwrap_fd_res = output_fd - data_fd_mod
            if losstype == 'l1':
                loss1 = (torch.sum(torch.abs(unwrap_fd_res * weightmask)))
            elif losstype == 'l2':
                loss1 = torch.sqrt(torch.sum((unwrap_fd_res * weightmask) ** 2))
            loss_list = [loss1]
            data_loss = loss1
        if torch.any(torch.isnan(data_loss)):
            break
        scaler.scale(data_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        rec = output_unwrap + Wop(wrapped_data - output_unwrap)
        rec = rec / torch.pi * venc

        loop.set_description(f'Epoch [{i}/{num_iter}]')
        if gt is not None and segmask is not None:
            nn = NRMSE(rec, gt, segmask).detach().cpu().numpy()
            if 1 - nn > bestn:
                bestn = 1 - nn
                bestepoch = i
            nrmse.append(1 - nn)

    ans = torch.zeros(orishape).to(device)
    ans[...,  minx:maxx, miny:maxy, minz:maxz] = rec[..., pads[-1][0]:SPE-pads[-1][-1]]
    # ans[0][weightmask[0] < 0.002] = ans[0][weightmask[0] < 0.002] 
    return ans
