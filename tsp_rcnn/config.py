# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_troi_config(cfg):
    """
    Add config for Transformer-ROI.
    """
    cfg.MODEL.SYNC_BN = True  # Deactivate SyncBatchNorm with single GPU
    cfg.MODEL.REINITIALIZE_BOX_PREDICTOR = False
    cfg.MODEL.ROI_BOX_HEAD.USE_COSINE = False

    cfg.MODEL.RPN.NUM_CONV = 1
    cfg.MODEL.FPN.NUM_REPEATS = 2

    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.MODEL.ROI_HEADS.SOFT_NMS_METHOD = "linear"
    cfg.MODEL.ROI_HEADS.SOFT_NMS_SIGMA = 0.5
    cfg.MODEL.ROI_HEADS.SOFT_NMS_PRUNE = 0.001
    cfg.MODEL.ROI_HEADS.TTA_NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.TTA_SCORE_THRESH_TEST = 0.001
    cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_ENABLED = False
    cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_METHOD = "linear"
    cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_SIGMA = 0.5
    cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_PRUNE = 0.001

    cfg.MODEL.MY_ROI_BOX_HEAD = CN()
    cfg.MODEL.MY_ROI_BOX_HEAD.D_MODEL = 512
    cfg.MODEL.MY_ROI_BOX_HEAD.NHEAD = 8
    cfg.MODEL.MY_ROI_BOX_HEAD.NUM_ENCODER_LAYERS = 6
    cfg.MODEL.MY_ROI_BOX_HEAD.NUM_DECODER_LAYERS = 6
    cfg.MODEL.MY_ROI_BOX_HEAD.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MY_ROI_BOX_HEAD.DROPOUT = 0.1
    cfg.MODEL.MY_ROI_BOX_HEAD.ACTIVATION = "relu"
    cfg.MODEL.MY_ROI_BOX_HEAD.NORMALIZE_BEFORE = True
    cfg.MODEL.MY_ROI_BOX_HEAD.USE_ENCODER_DECODER = False
    cfg.MODEL.MY_ROI_BOX_HEAD.USE_POSITION_ENCODING = False
    cfg.MODEL.MY_ROI_BOX_HEAD.USE_LINEAR_ATTENTION = False
    cfg.MODEL.MY_ROI_BOX_HEAD.NUM_FC = 1
    cfg.MODEL.MY_ROI_BOX_HEAD.FC_DIM = 1024
    cfg.MODEL.MY_ROI_BOX_HEAD.NUM_CONV = 0
    cfg.MODEL.MY_ROI_BOX_HEAD.CONV_DIM = 256
    cfg.MODEL.MY_ROI_BOX_HEAD.NUM_SELF_ATTENTION = 0
    cfg.MODEL.MY_ROI_BOX_HEAD.SELF_ATTENTION_DIM = 256

    cfg.MODEL.ROI_BOX_HEAD.ENCODER_FEATURE = "p5"
    cfg.MODEL.ROI_BOX_HEAD.EOS_COEF = 0.1
    cfg.MODEL.ROI_BOX_HEAD.ADD_NOISE_TO_PROPOSALS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_OBJ_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.L1_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.GIOU_WEIGHT = 2.0
    cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE = False
    cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE_UPPER_BOUND = 1.0
    cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE_LOWER_BOUND = 0.8
    cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP = False
    cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP_UPPER_BOUND = 1.0
    cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP_LOWER_BOUND = 0.8
    cfg.MODEL.ROI_BOX_HEAD.MAX_PROPOSAL_PER_BATCH = 0
    cfg.MODEL.ROI_BOX_HEAD.SEPARATE_OBJ_CLS = False
    cfg.MODEL.ROI_BOX_HEAD.FINETUNE_ON_SET = False
    cfg.MODEL.ROI_BOX_HEAD.CLS_HEAD_NO_BG = False
    cfg.MODEL.ROI_BOX_HEAD.DETR_EVAL_PROTOCOL = False
    cfg.MODEL.ROI_BOX_HEAD.USE_DETR_LOSS = False

    cfg.SOLVER.BOTTOM_UP_MULTIPLIER = 1.0
    cfg.SOLVER.TRANSFORMER_MULTIPLIER = 1.0
    cfg.SOLVER.OPTIMIZER = "SGD"
