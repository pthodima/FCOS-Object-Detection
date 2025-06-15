# FCOS Object Detection

## Project Overview
This project implements the FCOS (Fully Convolutional One-Stage) object detection model, which is a fully convolutional approach to object detection that eliminates the need for anchor boxes. The implementation is based on the paper "FCOS: Fully Convolutional One-Stage Object Detection" by Tian et al.

## Key Features
- Anchor-free object detection
- Fully convolutional architecture
- Center-ness branch for suppressing low-quality detections
- Multi-scale feature fusion
- Efficient training and inference

## System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 20GB+ free disk space

## Installation

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

Trained models and evaluation results can be found in the `results` directory.

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