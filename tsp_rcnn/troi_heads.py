# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
from typing import Dict, List, Optional, Tuple, Union
import torch
import copy
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals, add_ground_truth_to_proposals_single_image
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints, ROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from .mypooler import MyROIPooler
from .my_fast_rcnn_output import MyFastRCNNOutputLayers

__all__ = ["TransformerROIHeads"]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def add_noise_to_boxes(boxes):
    cxcy_boxes = box_xyxy_to_cxcywh(boxes)
    resize_factor = torch.rand(cxcy_boxes.shape, device=cxcy_boxes.device)
    new_cxcy = cxcy_boxes[..., :2] + cxcy_boxes[..., 2:] * (resize_factor[..., :2] - 0.5) * 0.2
    assert (cxcy_boxes[..., 2:] > 0).all().item()
    new_wh = cxcy_boxes[..., 2:] * (0.8 ** (resize_factor[..., 2:] * 2 - 1))
    assert (new_wh > 0).all().item()
    new_cxcy_boxes = torch.cat([new_cxcy, new_wh], dim=-1)
    new_boxes = box_cxcywh_to_xyxy(new_cxcy_boxes)
    return new_boxes


@ROI_HEADS_REGISTRY.register()
class TransformerROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: MyROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[MyROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[MyROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        add_noise_to_proposals: bool = False,
        encoder_feature: Optional[str] = None,
        random_sample_size: bool = False,
        random_sample_size_upper_bound: float = 1.0,
        random_sample_size_lower_bound: float = 0.8,
        random_proposal_drop: bool = False,
        random_proposal_drop_upper_bound: float = 1.0,
        random_proposal_drop_lower_bound: float = 0.8,
        max_proposal_per_batch: int = 0,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes
        self.add_noise_to_proposals = add_noise_to_proposals
        self.encoder_feature = encoder_feature
        self.random_sample_size = random_sample_size
        self.random_proposal_drop = random_proposal_drop
        self.max_proposal_per_batch = max_proposal_per_batch
        self.random_proposal_drop_upper_bound = random_proposal_drop_upper_bound
        self.random_proposal_drop_lower_bound = random_proposal_drop_lower_bound
        self.random_sample_size_upper_bound = random_sample_size_upper_bound
        self.random_sample_size_lower_bound = random_sample_size_lower_bound

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret["add_noise_to_proposals"] = cfg.MODEL.ROI_BOX_HEAD.ADD_NOISE_TO_PROPOSALS
        ret["encoder_feature"] = cfg.MODEL.ROI_BOX_HEAD.ENCODER_FEATURE
        ret["random_sample_size"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE
        ret["random_sample_size_upper_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE_UPPER_BOUND
        ret["random_sample_size_lower_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE_LOWER_BOUND
        ret["random_proposal_drop"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP
        ret["random_proposal_drop_upper_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP_UPPER_BOUND
        ret["random_proposal_drop_lower_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP_LOWER_BOUND
        ret["max_proposal_per_batch"] = cfg.MODEL.ROI_BOX_HEAD.MAX_PROPOSAL_PER_BATCH
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        ret["proposal_matcher"] = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = MyROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = MyFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        else:
            raise NotImplementedError

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        else:
            raise NotImplementedError

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features, proposals, targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None, return_box_features: bool=False
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = [features[f] for f in self.box_in_features]
        padded_box_features, dec_mask, inds_to_padded_inds = (
            self.box_pooler(box_features, [x.proposal_boxes for x in proposals]))
        enc_feature = None
        enc_mask = None
        if self.box_head.use_encoder_decoder:
            enc_feature = features[self.encoder_feature]
            b = len(proposals)
            h = max([x.image_size[0] for x in proposals])
            w = max([x.image_size[1] for x in proposals])
            enc_mask = torch.ones((b, h, w), dtype=torch.bool, device=padded_box_features.device)
            for c, image_size in enumerate([x.image_size for x in proposals]):
                enc_mask[c, :image_size[0], :image_size[1]] = False
            names = ["res1", "res2", "res3", "res4", "res5"]
            if self.encoder_feature == "p6":
                names.append("p6")
            for name in names:
                if name == "res1":
                    target_shape = ((h+1)//2, (w+1)//2)
                else:
                    x = features[name]
                    target_shape = x.shape[-2:]
                m = enc_mask
                enc_mask = F.interpolate(m[None].float(), size=target_shape).to(torch.bool)[0]

        max_num_proposals = padded_box_features.shape[1]
        normalized_proposals = []
        for x in proposals:
            gt_box = x.proposal_boxes.tensor
            img_h, img_w = x.image_size
            gt_box = gt_box / torch.tensor([img_w, img_h, img_w, img_h],
                                           dtype=torch.float32, device=gt_box.device)
            gt_box = torch.cat([box_xyxy_to_cxcywh(gt_box), gt_box], dim=-1)
            gt_box = F.pad(gt_box, [0, 0, 0, max_num_proposals - gt_box.shape[0]])
            normalized_proposals.append(gt_box)
        normalized_proposals = torch.stack(normalized_proposals, dim=0)

        padded_box_features = self.box_head(enc_feature, enc_mask, padded_box_features, dec_mask, normalized_proposals)
        box_features = padded_box_features[inds_to_padded_inds]
        predictions = self.box_predictor(box_features)

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, targets)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            if return_box_features:
                return losses, box_features
            else:
                return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances
        else:
            raise NotImplementedError

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances
        else:
            raise NotImplementedError

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [copy.deepcopy(x.gt_boxes) for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []

        for proposals_per_image, targets_per_image, gt_boxes_per_image in zip(proposals, targets, gt_boxes):
            has_gt = len(targets_per_image) > 0

            if self.add_noise_to_proposals:
                proposals_per_image.proposal_boxes.tensor = (
                    add_noise_to_boxes(proposals_per_image.proposal_boxes.tensor))

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            if not torch.any(matched_labels == 1) and self.proposal_append_gt:
                gt_boxes_per_image.tensor = add_noise_to_boxes(gt_boxes_per_image.tensor)
                proposals_per_image = add_ground_truth_to_proposals_single_image(gt_boxes_per_image,
                                                                                 proposals_per_image)

                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                proposals_per_image.set('gt_idxs', sampled_targets)
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                proposals_per_image.set('gt_idxs', torch.zeros_like(sampled_idxs))

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        if self.random_sample_size:
            diff = self.random_sample_size_upper_bound - self.random_sample_size_lower_bound
            sample_factor = self.random_sample_size_upper_bound - np.random.rand(1)[0] * diff
            nms_topk = int(matched_idxs.shape[0] * sample_factor)
            matched_idxs = matched_idxs[:nms_topk]
            matched_labels = matched_labels[:nms_topk]

        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        if self.random_proposal_drop:
            diff = self.random_proposal_drop_upper_bound - self.random_proposal_drop_lower_bound
            sample_factor = self.random_proposal_drop_upper_bound - np.random.rand(1)[0] * diff
            nms_topk = int(sampled_idxs.shape[0] * sample_factor)
            subsample_idxs = np.random.choice(sampled_idxs.shape[0], nms_topk, replace=False)
            subsample_idxs = torch.from_numpy(subsample_idxs).to(sampled_idxs.device)
            sampled_idxs = sampled_idxs[subsample_idxs]

        return sampled_idxs, gt_classes[sampled_idxs]
