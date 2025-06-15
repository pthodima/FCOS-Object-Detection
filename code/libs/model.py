import math
import operator
from itertools import accumulate
from typing import Dict, List

import torch
import torchvision
from torch.nn.functional import binary_cross_entropy_with_logits, binary_cross_entropy

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        # Apply convolutions to all feature pyramid levels
        outputs = []
        for feature in x:  # x is a list of feature maps at each pyramid level
            out = self.conv(feature)
            cls_logit = self.cls_logits(out)

            #Permute the classification logits to the shape [N, H, W, C]
            cls_logit = cls_logit.permute(0, 2, 3, 1)

            outputs.append(cls_logit)
        return outputs  # List of tensors, one for each pyramid level

class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        # Apply convolutions to all feature pyramid levels
        bbox_outputs = []
        ctr_outputs = []
        for feature in x:  # x is a list of feature maps at each pyramid level
            out = self.conv(feature)
            bbox_reg = self.bbox_reg(out)
            ctrness = self.bbox_ctrness(out)

            bbox_reg = bbox_reg.permute(0, 2, 3, 1)
            ctrness = ctrness.permute(0, 2, 3, 1)

            bbox_outputs.append(bbox_reg)
            ctr_outputs.append(ctrness)
        return bbox_outputs, ctr_outputs  # Two lists of tensors

class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        # images: List[tensor] each of shape tensor.Size([3, h, w])
        # targets: List[Dict] with each dict having keys 'boxes': tensor.Size({num_boxes, 4}),
        #                                                'labels': tensor.Size([num_boxes]),
        #                                                'image_id': tensor.Size([1]),
        #                                                'area': tensor.Size([numBoxes]),
        #                                                'iscrowd': tensor.Size([numBoxes]) #ToDo: What is 'iscrowd' here?
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:] # [C, H, W]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Here we will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """
    def compute_loss(self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits, lambda_reg=1):
        # print("target_boxes.shape: ", [t['boxes'].shape for t in targets])
        # print("target_area.shape: ", [t['area'].shape for t in targets])
        # print("target_labels.shape: ", [t['labels'].shape for t in targets])
        # print("points.shape: ", [p.shape for p in points])
        # print("strides.shape: ", strides.shape)
        # print("reg_range.shape: ", reg_range.shape)
        # print("cls_logits.shape: ", [c.shape for c in cls_logits])
        # print("reg_outputs.shape: ", [r.shape for r in reg_outputs])
        # print("ctr_logits.shape: ", [c.shape for c in ctr_logits])

        num_fpn_levels = strides.shape[0]
        batch_size = len(targets)
        level_flat_points = [pl.reshape([-1, 2]) for pl in points] # [h_level * w_level, 2] each
        labels, reg_targets = self.prepare_targets(targets, level_flat_points, reg_range, strides)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        points_over_batch = []
        strides_over_batch = []
        for l in range(len(labels)): # looping over levels?
            box_cls_flatten.append(cls_logits[l].reshape(-1, self.num_classes))
            box_regression_flatten.append(reg_outputs[l].reshape(-1, 4))
            centerness_flatten.append(ctr_logits[l].reshape(-1))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            points_over_batch.append(level_flat_points[l].repeat(batch_size, 1))
            strides_over_batch.append(strides[l].repeat(batch_size * len(level_flat_points[l])))

        box_cls_flatten = torch.cat(box_cls_flatten)
        box_regression_flatten = torch.cat(box_regression_flatten)
        centerness_flatten = torch.cat(centerness_flatten)
        labels_flatten = torch.cat(labels_flatten)
        reg_targets_flatten = torch.cat(reg_targets_flatten)
        points_over_batch = torch.cat(points_over_batch)
        strides_over_batch = torch.cat(strides_over_batch)

        pos_inds = torch.nonzero(labels_flatten != self.num_classes).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        pos_points_over_batch = points_over_batch[pos_inds]
        strides_over_batch = strides_over_batch[pos_inds]

        cls_loss = sigmoid_focal_loss(box_cls_flatten, self.convert_labels_one_hot(labels_flatten), alpha=0.25, reduction="sum")

        num_pos = pos_inds.numel()
        if num_pos > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            target_boxes = self.get_boxes_from_ltrb(
                reg_targets_flatten,
                pos_points_over_batch[:, 1],
                pos_points_over_batch[:, 0],
                strides_over_batch
            )
            pred_boxes = self.get_boxes_from_ltrb(
                box_regression_flatten,
                pos_points_over_batch[:, 1],
                pos_points_over_batch[:, 0],
                strides_over_batch
            )

            reg_loss = giou_loss(pred_boxes, target_boxes, reduction="mean")

            ctr_loss = binary_cross_entropy_with_logits(centerness_flatten, centerness_targets)

            cls_loss = cls_loss / num_pos

        else:
            reg_loss = box_regression_flatten.sum()
            ctr_loss = centerness_flatten.sum()

        final_loss = cls_loss + lambda_reg * (ctr_loss + reg_loss)
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss
        }

    def prepare_targets(self, targets, points, reg_range, strides):
        expanded_reg_range = []
        for l, points_per_level in enumerate(points):
            # object_sizes_of_interest_per_level = \
            #     points_per_level.new_tensor(reg_range[l])
            expanded_reg_range.append(
                # object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
                reg_range[l][None].expand(len(points_per_level), -1)
            )

        expanded_reg_range = torch.cat(expanded_reg_range)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points)
        labels, reg_targets = self.compute_targets_per_location(
            targets, points_all_level, strides, expanded_reg_range, num_points_per_level
        )

        for i in range(len(labels)): # looping over images
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            # if self.norm_reg_targets:
            reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_per_location(self, targets, points, strides, reg_ranges, num_points_per_level):
        labels = []
        reg_targets = []
        xs, ys = points[:, 1], points[:, 0]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im['boxes']
            labels_per_im = targets_per_im['labels']
            area = targets_per_im['area']

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2) # [num_points, num_gt_boxes, 4]

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            if self.center_sampling_radius > 0:
                is_in_boxes = is_in_boxes & self.get_sample_region(
                    bboxes,
                    strides,
                    num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )

            max_reg_targets_per_img = reg_targets_per_im.max(dim=2)[0]
            is_in_reg_range = \
                (max_reg_targets_per_img >= reg_ranges[:, [0]]) & \
                (max_reg_targets_per_img <= reg_ranges[:, [1]])

            points_to_gt_area = area[None].repeat(len(points), 1)
            points_to_gt_area[is_in_boxes == 0] = 1e10
            points_to_gt_area[is_in_reg_range == 0] = 1e10

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            points_to_min_area, points_to_gt_inds = points_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(points)), points_to_gt_inds]
            labels_per_im = labels_per_im[points_to_gt_inds]
            labels_per_im[points_to_min_area == 1e10] = self.num_classes # background class

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def get_sample_region(self, gt, strides, num_points_per_level, gt_xs, gt_ys, radius=1.0):
        """
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        """
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        ctr_x = (gt[..., 0] + gt[..., 2]) / 2
        ctr_y = (gt[..., 1] + gt[..., 3]) / 2
        ctr_gt = gt.new_zeros(gt.shape)

        # no gt boxes
        if ctr_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape)
        beg = 0
        for level, n_p in enumerate(num_points_per_level):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = ctr_x[beg:end] - stride
            ymin = ctr_y[beg:end] - stride
            xmax = ctr_x[beg:end] + stride
            ymax = ctr_y[beg:end] + stride
            # limit sample region in gt
            ctr_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            ctr_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            ctr_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2], xmax, gt[beg:end, :, 2]
            )
            ctr_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3], ymax, gt[beg:end, :, 3]
            )
            beg = end
        left = gt_xs[:, None]  - ctr_gt[..., 0]
        right = ctr_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - ctr_gt[..., 1]
        bottom = ctr_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def convert_labels_one_hot(self, labels):
        return F.one_hot(labels, self.num_classes+1)[:, :-1]

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def get_boxes_from_ltrb(self, ltrb, x, y, strides):
        l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
        x1 = x - l * strides
        y1 = y - t * strides
        x2 = x + r * strides
        y2 = y + b * strides
        return torch.stack([x1, y1, x2, y2], dim=1)


    """
    The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """
    def inference(self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes):
        num_imgs = len(image_shapes)
        num_classes = cls_logits[0].shape[-1]

        results: List[Dict[str, Tensor]] = []

        for img_index in range(num_imgs):
            box_reg_img = [r[img_index] for r in reg_outputs] # [H, W, 4] each
            cls_logits_img = [c[img_index] for c in cls_logits] # [H, W, C] each
            ctr_logits_img = [c[img_index] for c in ctr_logits] # [H, W, 1] each

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_reg_per_level, cls_logits_per_level, ctr_logits_per_level, points_per_level, stride in zip(box_reg_img, cls_logits_img, ctr_logits_img, points, strides):
                scores_per_level = torch.sqrt(
                    torch.sigmoid(cls_logits_per_level) * torch.sigmoid(ctr_logits_per_level)
                ).flatten()
                points_per_level = points_per_level.view(-1, 2)

                score_pass_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[score_pass_idxs]
                topk_idxs = torch.where(score_pass_idxs)[0]

                num_topk = min(score_pass_idxs.sum(), self.topk_candidates)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                pos_points = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                pos_reg_lvl = box_reg_per_level.view(-1, 4)[pos_points]
                pos_points_lvl = points_per_level[pos_points]

                # Identify actual boxes
                boxes_lvl = torch.zeros_like(pos_reg_lvl)
                boxes_lvl[:, 0] = pos_points_lvl[:, 1] - pos_reg_lvl[:, 0] * stride # l -> x1
                boxes_lvl[:, 1] = pos_points_lvl[:, 0] - pos_reg_lvl[:, 1] * stride # t -> y1
                boxes_lvl[:, 2] = pos_points_lvl[:, 1] + pos_reg_lvl[:, 2] * stride # r -> x2
                boxes_lvl[:, 3] = pos_points_lvl[:, 0] + pos_reg_lvl[:, 3] * stride # b -> y2

                # Clip predictions to image size
                h, w = image_shapes[img_index]
                boxes_lvl[:, 0].clamp_(min=0, max=w) # x1
                boxes_lvl[:, 1].clamp_(min=0, max=h) # y1
                boxes_lvl[:, 2].clamp_(min=0, max=w) # x2
                boxes_lvl[:, 3].clamp_(min=0, max=h) # y2

                image_boxes.append(boxes_lvl)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            final_boxes = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            final_boxes = final_boxes[:self.detections_per_img]

            results.append(
                {
                    "boxes": image_boxes[final_boxes],
                    "scores": image_scores[final_boxes],
                    "labels": image_labels[final_boxes] + 1,
                }
            )
        return results