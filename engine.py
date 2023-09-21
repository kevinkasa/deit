# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets, sample_ids = batch
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_ids = sample_ids.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        ### AUM ###
        # Use torch.distributed.all_gather to collect tensors from all devices
        all_outputs_gathered = [torch.empty_like(outputs) for _ in range(torch.distributed.get_world_size())]
        all_targets_gathered = [torch.empty_like(targets) for _ in range(torch.distributed.get_world_size())]
        all_sample_ids_gathered = [torch.empty_like(sample_ids) for _ in range(torch.distributed.get_world_size())] # [tensor(batch_size), nGPUs] 
        torch.distributed.barrier()

        torch.distributed.all_gather(all_outputs_gathered, outputs)
        torch.distributed.all_gather(all_targets_gathered, targets)
        torch.distributed.all_gather(all_sample_ids_gathered, sample_ids)
        torch.distributed.barrier()

        # Concatenate the gathered tensors from all devices into one tensor
        all_outputs_gathered = torch.cat(all_outputs_gathered, dim=0)
        all_targets_gathered = torch.cat(all_targets_gathered, dim=0)
        all_sample_ids_gathered = torch.cat(all_sample_ids_gathered, dim=0)
        all_sample_ids_gathered = all_sample_ids_gathered.tolist()

        if args.rank == 0:
            records = args.aum_calculator.update(all_outputs_gathered, all_targets_gathered, all_sample_ids_gathered)        

        ### /AUM ###

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, save_output = False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    #for param in model.parameters():
    #        print(param.requires_grad)
    #        param.requires_grad = False


    if save_output:
        len_data = len(data_loader.dataset)
        outputs = np.ones((len_data, args.nb_classes))
        targets = np.ones((len_data, ))
        counter = 0

        logits_list = []
        labels_list = []
        # ece_criterion = utils.ECELoss().cuda()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target, sample_ids = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sample_ids = sample_ids.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        if save_output:
            outputs[counter:counter+images.shape[0],:] = output.softmax(dim=1).cpu().numpy()
            targets[counter:counter+target.shape[0]] = target.cpu().numpy().astype(int)
            counter += images.shape[0]

            logits_list.append(output.softmax(dim=1).cpu().numpy())
            labels_list.append(target.cpu().numpy().astype(int))


        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if save_output:
        np.savez(args.output_dir / 'outputs.npz', smx = outputs, labels = targets)
        # np.save(args.output_dir / 'outputs.npy', outputs)
        # np.save(args.output_dir / 'targets.npy', targets)
        #logits = np.concatenate(logits_list, axis=0)
        #labels = np.concatenate(labels_list, axis=0)
        # ece = ece_criterion(logits, labels).item()
        # print('Before temperature ECE: %.3f' % (ece))
        #np.save(args.output_dir / 'logits.npy', logits, )
        #np.save(args.output_dir / 'labels.npy', labels)

    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
