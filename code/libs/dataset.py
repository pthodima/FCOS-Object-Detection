import os
import random
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .transforms import Compose, ConvertAnnotations, RandomHorizontalFlip, ToTensor


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))


def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class VOCDetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load VOC data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def get_cls_names(self):
        cls_names = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        return cls_names

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class COCODetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load COCO data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def get_cls_names(self):
        # COCO class names
        # this should be doable with coco.getCatIds()
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        # for t in target:
        #     t['area'] = t['bbox'][2] * t['bbox'][3] # TODO: Change area of image to match what we did for VOC, Although this shouldn't be necessary
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_dataset(name, split, img_folder, json_folder):
    """
    Create VOC dataset with default transforms for training / inference.
    New datasets can be linked here.
    """
    if name == "VOC2007":
        assert split in ["trainval", "test"]
        is_training = split == "trainval"
    elif name == "COCO":
        assert split in ["train", "val"]
        is_training = split == "val"
    else:
        print("Unsupported dataset")
        return None

    if is_training:
        transforms = Compose([ConvertAnnotations(), RandomHorizontalFlip(), ToTensor()])
    else:
        transforms = Compose([ConvertAnnotations(), ToTensor()])

    if name == "VOC2007":
        dataset = VOCDetection(
            img_folder, os.path.join(json_folder, split + ".json"), transforms
        )
    elif name == "COCO":
        dataset = COCODetection(
            os.path.join(img_folder, split+"2017"),
            os.path.join(json_folder, "instances_"+split+"2017.json"),
            transforms
        )
    return dataset


def build_dataloader(dataset, is_training, batch_size, num_workers):
    """
    Create a dataloader for datasets
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=True,
    )
    return loader
