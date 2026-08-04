# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import random
import math

import timesformer.models.losses as losses
import timesformer.models.optimizer as optim
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TrainMeter, ValMeter, UnlabelMeter
from timesformer.utils.multigrid import MultigridSchedule
from timesformer.utils.ema import ModelEma
from .asn_utils import consistency_loss_asn, kmeans
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from thop import profile

import os
import numpy as np

logger = logging.get_logger(__name__)


class TubeMasking:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def tubemask(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1))
        return mask


def train_epoch(
        train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size

    for cur_iter, (inputs, _, labels, _, meta, _) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")


        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
        else:
            preds = model(inputs)

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def temporal_aug(strong):
    strong_temporal = strong
    index = torch.randperm(strong_temporal.shape[2])
    strong_temporal = strong[:, :, index, :, :]
    return strong_temporal


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_mask(max_prob, max_std):
    mask_prob = 2 * torch.sigmoid((torch.exp(max_prob) - math.exp(1)))
    mask_std = 2 * (torch.sigmoid((1 / (20 * max_std + 1e-9))) - 0.5)
    mask = mask_prob * mask_std
    return mask


def ssl_train_epoch(
        train_loader, unlabel_loader, model, model_ema, optimizer, unlabel_meter, cur_epoch, cfg,
        asn_state, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Enable train mode.
    model.train()
    unlabel_meter.iter_tic()
    data_size = len(unlabel_loader)

    # Enable dropout of EMA teacher.
    enable_dropout(model_ema.ema)

    num_frames = cfg.DATA.SAMPLE_FRAMES
    input_size = cfg.DATA.TRAIN_CROP_SIZE
    patch_size = (16, 16)
    window_size = [(num_frames[i], input_size // patch_size[0], input_size // patch_size[1]) for i in
                   range(len(num_frames))]

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size

    train_iter = iter(train_loader)

    for cur_iter, (weak, strong, _, _, _, index) in enumerate(unlabel_loader):
        try:
            inputs, _, labels, _, meta, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, _, labels, _, meta, _ = next(train_iter)
        if cfg.NUM_GPUS:
            if isinstance(weak, (list,)):
                for i in range(len(weak)):
                    weak[i] = weak[i].cuda(non_blocking=True)
                    strong[i] = strong[i].cuda(non_blocking=True)
            else:
                weak = weak.cuda(non_blocking=True)
                strong = strong.cuda(non_blocking=True)

            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        unlabel_meter.data_toc()
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        loss_unlabel = torch.nn.CrossEntropyLoss(reduction="none")
        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
        else:
            preds = model(inputs)
        # Video reverse.
        strong_temporal = []
        for i in range(len(strong)):
            strong_tensor = strong[i].clone()
            if i < len(strong) - 1:
                # Reverse
                strong_t = strong_tensor.flip(dims=(1,))
                # Shuffle
                #strong_t = temporal_aug(strong_tensor)
            else:
                strong_t = strong_tensor.clone()
            strong_temporal.append(strong_t)
            strong[i] = F.dropout2d(strong_t, 0.2)
        student = model(strong)
        for i in range(len(strong)):
            idx = torch.randperm(strong[i].size(0))
            strong_resort = strong_temporal[i][idx, :]
            strong[i] = torch.cat([strong[i], strong_resort], dim=0)

        # Teacher predict for 10 times.
        model.eval()
        pseudo_label = []
        with torch.no_grad():
            for _ in range(10):
                if model_ema:
                    teacher = model_ema.ema(weak).detach()
                    pseudo_label.append(F.softmax(teacher, dim=-1))
                else:
                    teacher = model(weak).detach()
                    pseudo_label.append(F.softmax(teacher, dim=-1))

        pseudo_label = torch.stack(pseudo_label)
        probs_std = torch.std(pseudo_label, dim=0)
        pseudo_label = torch.mean(pseudo_label, dim=0)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        targets_u = max_idx
        max_std = probs_std.gather(1, targets_u.view(-1, 1)).squeeze(1)
        label_matrix = asn_state["label_matrix"]
        label_bank = asn_state["label_bank"]
        centroids = asn_state["centroids"]
        label_dics = asn_state["label_dics"]
        clusters = asn_state["clusters"]
        label_count = asn_state["label_count"]
        num_classes = cfg.ASN.NUM_CLASSES
        alpha = cfg.ASN.ALPHA
        C_lower = cfg.ASN.C_LOWER
        num_eval_iter = cfg.ASN.NUM_EVAL_ITER
        N = cfg.ASN.N
        centroids = centroids
        global_iter = cur_epoch * data_size + cur_iter
        it = global_iter
        print(global_iter,'global_iter')
        c_flag = True
        C = round(num_classes / alpha) + 1

        #print(index[:20], 'index')
        for i in range(len(index)):
            if not index[i].cpu().item() in label_bank.keys():
                label_bank[index[i].cpu().item()] = max_idx[i].cpu().item()
            else:
                if label_bank[index[i].cpu().item()] != max_idx[i].cpu().item():
                    label_matrix[label_bank[index[i].cpu().item()], max_idx[i].cpu().item(), label_count] = label_matrix[label_bank[index[i].cpu().item()],max_idx[i].cpu().item(), label_count] + 1
                    label_bank[index[i].cpu().item()] = max_idx[i].cpu().item()
        # current slot already used
        next_label_count = (label_count + 1) % N
        label_matrix[:, :, next_label_count].zero_()
        label_count = next_label_count
        ###############
        if cur_iter % 500 == 0:
            save_dir = os.path.join(cfg.OUTPUT_DIR, "asn_vis")
            os.makedirs(save_dir, exist_ok=True)

            # [K, K, N] -> [K, K]
            transition_matrix = torch.sum(label_matrix, dim=2)

            transition_np = transition_matrix.detach().cpu().numpy()
            #sim_np = sim_matrix.detach().cpu().numpy()

            base_name = f"epoch_{cur_epoch:03d}_iter_{cur_iter:05d}"


            # save txt files
            np.savetxt(
                os.path.join(save_dir, base_name + "_transition.txt"),
                transition_np,
                fmt="%.4f"
            )


        ######################
        # perform k-means
        if it % num_eval_iter == 0 and c_flag:
            for i in range(C):
                label_dics[i], clusters[i], centroids[i] = kmeans(
                    torch.sum(label_matrix, axis=2, keepdim=False).numpy(), i + C_lower + 1, centroids[i])
        # c_count = it % C
        # label_dics[c_count], clusters[c_count], centroids[c_count] = kmeans(
        #     torch.sum(label_matrix, axis=2, keepdim=False).numpy(), c_count + C_lower + 1, centroids[c_count])

        model.train()
        mask_ratio = np.random.beta(cfg.TRAIN.BETA, cfg.TRAIN.BETA)
        tubemask_list = []
        for i in range(len(window_size)):
            tube = TubeMasking(input_size=window_size[i], mask_ratio=mask_ratio)
            tubemask = tube.tubemask()
            tubemask = tubemask[None].repeat(teacher.size(0), axis=0)
            tubemask = torch.Tensor(tubemask).cuda()
            tubemask_list.append(tubemask)

        pseudo = torch.zeros(teacher.size(0), teacher.size(-1)).cuda()
        mask = get_mask(max_probs, max_std)
        for i in range(teacher.size(0)):
            pseudo[i] = pseudo_label[i]
        # Compute the loss.
        loss_un = (loss_unlabel(student, targets_u) * mask).mean()
        loss_label = loss_fun(preds, labels)
        cos_loss = consistency_loss_asn(pseudo_label, student, label_dics, clusters, alpha,
                                        num_classes)

        loss = loss_label + 0.5 * loss_un + 0.5 * cos_loss
        # check Nan Loss.
        misc.check_nan_losses(loss)

        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        if model_ema:
            model_ema.update(model)

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            unlabel_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                    [loss_label] = du.all_reduce([loss_label])
                    [loss_un] = du.all_reduce(loss_un)
                loss = loss.item()
                loss_un = loss_un.item()
                loss_label = loss_label.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, loss_label, loss_un, top1_err, top5_err = du.all_reduce(
                        [loss, loss_label, loss_un, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, loss_label, loss_un, top1_err, top5_err = (
                    loss.item(),
                    loss_label.item(),
                    loss_un.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            unlabel_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                loss_label,
                loss_un,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        unlabel_meter.iter_toc()  # measure allreduce for this meter
        unlabel_meter.log_iter_stats(cur_epoch, cur_iter)
        unlabel_meter.iter_tic()

    # Log epoch stats.
    unlabel_meter.log_epoch_stats(cur_epoch)
    #####################
    asn_state["label_matrix"] = label_matrix
    asn_state["label_bank"] = label_bank
    asn_state["centroids"] = centroids
    asn_state["label_dics"] = label_dics
    asn_state["clusters"] = clusters
    asn_state["label_count"] = label_count
    ########################
    unlabel_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, inputs_fast, labels, _, meta, _) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                    inputs_fast[i] = inputs_fast[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
                inputs_fast = inputs_fast.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
        start_epoch = 0
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    unlabel_loader = loader.construct_loader(cfg, "unlabel2")
    #unlabel_loader = loader.construct_loader(cfg, "unlabel")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    train_meter = TrainMeter(len(train_loader), cfg)
    unlabel_meter = UnlabelMeter(len(unlabel_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    # labeled training
    for cur_epoch in range(start_epoch, cfg.TRAIN.WARM_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )

        if (cur_epoch + 1) % 5 == 0:
            is_checkp_epoch = True
            is_eval_epoch = True
        else:
            is_checkp_epoch = False
            is_eval_epoch = False

        # Compute precise BN stats.
        if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    # set ema_model
    model_ema = ModelEma(model, decay=cfg.TRAIN.EMA)

    # ====== asn initial ======
    num_classes = cfg.ASN.NUM_CLASSES
    alpha = cfg.ASN.ALPHA
    C = round(num_classes / alpha) + 1
    C_lower = cfg.ASN.C_LOWER
    N = cfg.ASN.N
    asn_state = {
        "label_matrix": torch.zeros(num_classes, num_classes, N),
        "label_bank": {},
        "centroids": [
            np.random.choice(
                [i for i in range(num_classes)],
                i + C_lower + 1,
                False
            ).tolist()
            for i in range(C)
        ],
        "label_dics": [{} for _ in range(C)],
        "clusters": [[] for _ in range(C)],
        "label_count": 0
    }
    # unlabeled training
    for cur_epoch in range(cfg.TRAIN.WARM_EPOCH, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        loader.shuffle_dataset(unlabel_loader, cur_epoch)

        optimizer_ema = optim.construct_optimizer(model_ema.ema, cfg)

        # Train for one epoch.
        ssl_train_epoch(
            train_loader, unlabel_loader, model, model_ema, optimizer, optimizer_ema, unlabel_meter, cur_epoch, cfg,
             asn_state, writer
        )
        if (cur_epoch + 1) % 5 == 0:
            is_checkp_epoch = True
            is_eval_epoch = True
        else:
            is_checkp_epoch = False
            is_eval_epoch = False

        # Compute precise BN stats.
        if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            if model_ema is None:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
            else:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        # Evaluate the model on validation set.
        if is_eval_epoch:
            if model_ema is None:
                eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
            else:
                eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
