# DigitDetect: Object Localization and Detection in Augmented MNIST

Deep learning project for `INF265` at the University of Bergen exploring two computer vision tasks on an augmented MNIST-style dataset:

- `Object_localization.ipynb`: single-object localization and digit classification
- `Object_detection.ipynb`: multi-object detection with a grid-based, YOLO-inspired pipeline

The project studies how convolutional neural networks can predict object presence, class labels, and bounding boxes in noisy grayscale images where digits vary in position, size, and rotation.

## Project Overview

This repository contains two complementary pipelines:

1. `Object_localization.ipynb`
   Trains and evaluates CNN models that predict:
   - object presence `pc`
   - bounding box `(x, y, w, h)`
   - digit class `c`

2. `Object_detection.ipynb`
   Extends the task to grid-based object detection with multiple objects per image, including:
   - exploratory data analysis
   - grid-label handling
   - fully convolutional detection models
   - model selection and final evaluation workflow

The accompanying report is available in [Project2.pdf](Project2.pdf).

## Repository Structure

```text
.
|-- data/
|   |-- localization_*.pt
|   |-- detection_*.pt
|   `-- list_y_true_*.pt
|-- Object_localization.ipynb
|-- Object_detection.ipynb
|-- Project2.pdf
|-- requirements.txt
`-- README.md
```

## Dataset

The repository uses pre-generated PyTorch dataset files (`.pt`) stored in `data/`.

- Localization data:
  - `data/localization_train.pt`
  - `data/localization_val.pt`
  - `data/localization_test.pt`
- Detection data:
  - `data/detection_train.pt`
  - `data/detection_val.pt`
  - `data/detection_test.pt`
  - `data/list_y_true_train.pt`
  - `data/list_y_true_val.pt`
  - `data/list_y_true_test.pt`

The images are grayscale and appear to be based on augmented MNIST digits placed in larger noisy canvases.

## Methods

### Localization

The localization notebook includes:

- dataset loading and sanity checks
- input normalization
- multiple CNN variants (`V1_baseline`, `V2_deeper`, `V3_bn_drop`)
- a custom loss combining:
  - objectness loss
  - bounding-box regression loss
  - digit classification loss
- hyperparameter search over:
  - learning rate
  - weight decay
  - bounding-box loss weight
- evaluation using:
  - classification accuracy
  - IoU
  - overall score
  - mAP@0.5

### Detection

The detection notebook includes:

- reproducible training setup
- pixel and label distribution analysis
- grid-based detection targets
- fully convolutional detection models
- model selection
- final evaluation and visualization utilities

## Results

Saved notebook outputs in `Object_localization.ipynb` show the best localization model as:

- Model: `V2_deeper`
- Best `lambda_bbox`: `20`
- Best hyperparameters: `lr=0.001`, `weight_decay=0.0`

Test-set results for the localization task:

- Accuracy: `0.9209`
- IoU: `0.7127`
- Overall score: `0.8168`
- mAP@0.5: `0.8513`

Presence detection performance reported in the notebook:

- Overall presence accuracy: `0.9988`
- No-digit accuracy: `1.0000`
- Digit-present accuracy: `0.9987`

The detection notebook contains the full training and evaluation pipeline, but final saved metrics were not embedded in the notebook output currently stored in this repository.

## Installation

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies listed in `requirements.txt`:

- `torch`
- `torchvision`
- `torchmetrics`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`

## How To Run

Start Jupyter and open the notebooks:

```bash
jupyter notebook
```

Then run:

- [Object_localization.ipynb](Object_localization.ipynb)
- [Object_detection.ipynb](Object_detection.ipynb)

Note:

- The localization notebook loads files from `data/...`.
- The detection notebook should be checked for dataset paths before rerunning, since the stored notebook code references detection files without the `data/` prefix.

## Authors

- Andreas Wold Slogedal
- Markus Dybevold Simonsen

## Suggested GitHub Repository Name

If you want a cleaner public-facing name than `deep-learning-project2`, I would use:

`digitdetect-mnist-localization-detection`

Shorter alternatives:

- `digitdetect`
- `mnist-localization-detection`
- `inf265-object-detection`
