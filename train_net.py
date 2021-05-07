#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import random
import numpy as np
import torch
import time
import math
import logging
from collections import defaultdict


import pickle
from fvcore.common.file_io import PathManager

from collections import OrderedDict
from itertools import count
from typing import Any, Dict, List, Set
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from tsp_rcnn import add_troi_config, DetrDatasetMapper
from tsp_fcos import add_fcos_config
from detectron2.utils.events import EventStorage

from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling import GeneralizedRCNNWithTTA, DatasetMapperTTA
from tsp_rcnn.my_fast_rcnn_output import fast_rcnn_inference_single_image

# Register PASCAL datasets
from tsp_rcnn.fsdet_data.builtin import register_all_pascal_voc
#register_all_pascal_voc()

class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HybridOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("optimizer", "SGD")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if group["optimizer"] == "SGD":
                    weight_decay = group['weight_decay']
                    momentum = group['momentum']
                    dampening = group['dampening']

                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        d_p = buf
                    p.add_(d_p, alpha=-group['lr'])

                elif group["optimizer"] == "ADAMW":
                    # Perform stepweight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Perform optimization step
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    raise NotImplementedError

        return loss


class AdetCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectronCheckpointer`, but is able to convert models
    in AdelaiDet, such as LPF backbone.
    """
    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded


def append_gt_as_proposal(gt_instances):
    for instances in gt_instances:
        instances.proposal_boxes = instances.gt_boxes
        instances.gt_idxs = torch.arange(len(instances.gt_boxes))
    return gt_instances


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self.clip_norm_val = 0.0
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                self.clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE

        DefaultTrainer.__init__(self, cfg)

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        #self._detect_anomaly(losses, loss_dict)  # removed with new detectron2

        metrics_dict = loss_dict
        #metrics_dict["data_time"] = data_time
        self._trainer._write_metrics(metrics_dict, data_time)

        self.optimizer.zero_grad()
        losses.backward()
        if self.clip_norm_val > 0.0:
            clipped_params = []
            for name, module in self.model.named_modules():
                for key, value in module.named_parameters(recurse=False):
                    if "transformer" in name:
                        clipped_params.append(value)
            torch.nn.utils.clip_grad_norm_(clipped_params, self.clip_norm_val)
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for name, _ in model.named_modules():
            print(name)

        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                optimizer_name = "SGD"
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

                if "bottom_up" in name:
                    lr = lr * cfg.SOLVER.BOTTOM_UP_MULTIPLIER
                elif "transformer" in name:
                    lr = lr * cfg.SOLVER.TRANSFORMER_MULTIPLIER
                    optimizer_name = "ADAMW"

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "optimizer": optimizer_name}]

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
        elif optimizer_type == "HYBRID":
            optimizer = HybridOptimizer(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.CROP.ENABLED:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = MyGeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def initialize_from_support(trainer_self):

        class_means = defaultdict(list)
        class_activations = defaultdict(list)

        print('Computing support set centroids')

        # Make sure this doesn't break on multigpu
        # Disable default Collate function
        support_loader = torch.utils.data.DataLoader(trainer_self.data_loader.dataset.dataset, batch_size=trainer_self.data_loader.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x:x)

        with EventStorage() as storage:

            for i, batched_inputs in enumerate(support_loader):
            #for i, batched_inputs in enumerate(trainer_self.data_loader):

                print('Processed {} batches'.format(i))

                self = trainer_self.model
                images = self.preprocess_image(batched_inputs)
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                features = self.backbone(images.tensor)

                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
                proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
                # Average box deatures here
                gt_as_proposals = append_gt_as_proposal(gt_instances)
                losses, box_features = self.roi_heads._forward_box(features, gt_as_proposals, gt_instances, return_box_features=True)

                box_features_idx = 0
                for instances in gt_as_proposals:
                    for gt_class in instances.gt_classes:
                        category_id = gt_class.item()
                        activation = box_features[box_features_idx]
                        class_activations[category_id].append(activation.detach().cpu())
                        box_features_idx += 1

        for category_id in class_activations:
            class_activations[category_id] = torch.stack(class_activations[category_id])
            class_means[category_id] = class_activations[category_id].mean(dim=0)
            print('Category: #{}, shape: {}'.format(category_id, class_activations[category_id].size()))

        pass



class MyGeneralizedRCNNWithTTA(GeneralizedRCNNWithTTA):
    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__(cfg, model, tta_mapper, batch_size)
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, GeneralizedRCNN
        ), "TTA is only supported on GeneralizedRCNN. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            self.cfg.MODEL.ROI_HEADS.TTA_SCORE_THRESH_TEST,
            self.cfg.MODEL.ROI_HEADS.TTA_NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_ENABLED,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_METHOD,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_SIGMA,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_PRUNE,
        )

        return merged_instances


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_troi_config(cfg)
    add_fcos_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    print("Random Seed:", cfg.SEED)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    # if cfg.MODEL.WEIGHTS.startswith("detectron2://ImageNetPretrained"):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Few-shot: reset parameters (if not resuming)
    if cfg.MODEL.REINITIALIZE_BOX_PREDICTOR:
        assert args.resume == False, "few-shot does not support resuming"
        print('Reinitializing output box predictor')
        trainer.model.roi_heads.box_predictor.cls_score.reset_parameters()
        trainer.model.roi_heads.box_predictor.bbox_pred.layers[-1].reset_parameters()

        # Few-shot: initialize cls_score weights to average of support set
        trainer.initialize_from_support()

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
