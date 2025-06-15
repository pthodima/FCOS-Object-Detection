# FCOS Object Detection

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding FCOS](#understanding-fcos)
3. [Model Architecture](#model-architecture)
4. [Model Inference](#model-inference)
5. [Model Training](#model-training)
6. [Running on COCO Dataset](#running-on-coco-dataset)
7. [Installation & Setup](#installation--setup)
8. [Results](#results)
9. [Acknowledgments](#acknowledgments)
10. [Team Contributions](#team-contributions)

---

## Introduction
This project implements the FCOS (Fully Convolutional One-Stage) object detection model, a fully convolutional approach that eliminates the need for anchor boxes. The implementation is based on the paper "FCOS: Fully Convolutional One-Stage Object Detection" by Tian et al.

## Understanding FCOS
- **Backbone Output:** Feature maps at different levels (C3, C4, C5) from ResNet, capturing spatial features at varying resolutions.
- **FPN Output:** Five levels (P3–P7), generated via lateral connections and convolutions, enabling detection of objects at multiple scales.
- **Sample Assignment:** Points are positive if they lie within the ground truth box, the center region, and the regression range for their FPN level. All other points are negative.
- **Loss Functions:**
  - *Classification Loss:* Focal loss to address class imbalance.
  - *Regression Loss:* Generalized IoU (GIoU) loss for bounding box regression.
  - *Center-ness Loss:* Binary cross-entropy loss for center-ness prediction.
- **Inference & Post-processing:** Decoding predictions, confidence thresholding, bounding box reconstruction, and non-maximum suppression (NMS).

## Model Architecture
- **Classification Head:** Applies convolutions at each FPN level to produce class logits.
- **Regression Head:** Two branches for bounding box edges and centerness, with ReLU activations to ensure non-negative outputs.

## Model Inference
- **Decoding:** For each point, predicts distances to box edges, class probabilities, and centerness scores. Final confidence is computed as the geometric mean of classification and centerness scores.
- **Post-processing:**
  - Confidence thresholding
  - Bounding box reconstruction
  - Non-maximum suppression (NMS)
- **Efficiency:** Top-K selection for computational efficiency, class-wise NMS, and stride scaling during decoding.

## Model Training
- **Workflow:**
  1. Assign targets to all points based on location, regression range, and center-sampling.
  2. Flatten predictions and targets across FPN levels and batch.
  3. Compute classification, regression, and centerness losses.
  4. Combine losses for optimization.
- **Losses:**
  - *Classification Loss:* Sigmoid focal loss, normalized by positive points.
  - *Regression Loss:* GIoU loss, normalized by positive points.
  - *Center-ness Loss:* Binary cross-entropy, normalized by positive points.
  - *Final Loss:* Sum of all three losses (with λ=1).
- **Results & Curves:** Training and validation curves are available in the writeup and results directory.

## Running on COCO Dataset
- **Dataset Setup:** Use `download_coco.sh` to download and organize the COCO dataset in the `data` directory.
- **Training Details:**
  - Backbone: ResNet18
  - Epochs: 4
  - Batch size: 32
  - Workers: 2
- **Special Handling:**
  - Remapping of missing label IDs to match COCO's 80-class setup.
  - Handling images with no ground truth boxes.
- **Results & Visualizations:** See results and images in the results directory and writeup.

## Installation & Setup

### Option 1: Using venv (Python's built-in virtual environment)

1. Create a new virtual environment:
```bash
python -m venv fcos_env
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source fcos_env/bin/activate
```
- On Windows:
```bash
fcos_env\Scripts\activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. To deactivate the virtual environment when you're done:
```bash
deactivate
```

### Option 2: Using Conda

1. Create a new conda environment:
```bash
conda create -n fcos_env python=3.8
```

2. Activate the conda environment:
```bash
conda activate fcos_env
```

3. Install PyTorch using conda (recommended):
```bash
conda install pytorch torchvision -c pytorch
```

4. Install the remaining requirements:
```bash
pip install -r requirements.txt
```

5. To deactivate the conda environment when you're done:
```bash
conda deactivate
```

### Verification
To verify the installation, run:
```bash
python -c "import torch; print(torch.__version__)"
```

## Dataset Preparation

### VOC2007
1. Download the VOC2007 dataset:
```bash
./download_dataset.sh
```

### COCO
1. Download the COCO dataset:
```bash
./download_coco.sh
```

## Usage

### Training

To train the model, use the following command:
```bash
python code/train.py path/to/config.yaml
```

Example configuration files can be found in the `code/configs` directory.

### Evaluation

To evaluate a trained model:
```bash
python code/eval.py path/to/config.yaml path/to/checkpoint
```

Optional arguments:
- `-e, --epoch`: Specify checkpoint epoch (default: -1, uses latest)
- `-p, --print-freq`: Print frequency (default: 10 iterations)
- `-v, --viz`: Visualize detection results

## Model Architecture

The FCOS model consists of the following main components:
1. Backbone network (ResNet variants)
2. Feature Pyramid Network (FPN)
3. FCOS Head (Classification and Regression)
4. Post-processing (NMS)

## Configuration

The model can be configured using YAML files. Key configuration parameters include:
- Backbone network type
- Input image size
- Feature pyramid strides
- Training parameters (learning rate, batch size, etc.)
- Test parameters (score threshold, NMS threshold, etc.)

## Results

### COCO Training (ResNet18, 4 epochs)

| Metric            | All   | Small | Medium | Large |
|-------------------|-------|-------|--------|-------|
| **AP@[.5:.95]**   | 0.108 | 0.053 | 0.114  | 0.162 |
| **AP@0.5**        | 0.195 |   –   |   –    |   –   |
| **AP@0.75**       | 0.106 |   –   |   –    |   –   |
| **AR@1**          | 0.152 |   –   |   –    |   –   |
| **AR@10**         | 0.245 |   –   |   –    |   –   |
| **AR@100**        | 0.264 | 0.113 | 0.279  | 0.388 |

---

### Pretrained Inference

| Metric            | All   | Small | Medium | Large |
|-------------------|-------|-------|--------|-------|
| **AP@[.5:.95]**   | 0.330 | 0.078 | 0.202  | 0.406 |
| **AP@0.5**        | 0.609 |   –   |   –    |   –   |
| **AP@0.75**       | 0.322 |   –   |   –    |   –   |
| **AR@1**          | 0.318 |   –   |   –    |   –   |
| **AR@10**         | 0.498 |   –   |   –    |   –   |
| **AR@100**        | 0.537 | 0.203 | 0.440  | 0.617 |

---

### Training with Pretrained Config

| Metric            | All   | Small | Medium | Large |
|-------------------|-------|-------|--------|-------|
| **AP@[.5:.95]**   | 0.379 | 0.060 | 0.228  | 0.477 |
| **AP@0.5**        | 0.647 |   –   |   –    |   –   |
| **AP@0.75**       | 0.389 |   –   |   –    |   –   |
| **AR@1**          | 0.348 |   –   |   –    |   –   |
| **AR@10**         | 0.517 |   –   |   –    |   –   |
| **AR@100**        | 0.545 | 0.172 | 0.431  | 0.641 |

---

**Legend:**
- AP = Average Precision
- AR = Average Recall
- "–" = Not reported for that area

## Citation

If you use FCOS in your research, please cite:
```
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:2006.09214},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The loss computation is inspired by the official FCOS implementation: https://github.com/tianzhi0549/FCOS/tree/master/fcos_core/modeling/rpn/fcos
- Some text was rephrased using chatgpt.com and code assistance from claude.ai.

## Team Contributions
- **Ryan Usher:** Inference code and Understanding FCOS section.
- **Rithik Jain:** Loss computation and COCO training.
- **Pavan Thodima:** Inference, loss computation, and ResNet50 training on VOC dataset. 