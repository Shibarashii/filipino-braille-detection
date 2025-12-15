# Braille Detection Model Training Guide

This guide covers training and evaluating the YOLO-based Braille detection model for Filipino Grade 1 Braille.

## Overview

The training pipeline uses YOLOv8 for object detection to identify individual Braille characters, including:

- Standard Braille letters (a-z)
- Numbers (0-9)
- Special characters (capital indicator, number indicator, enye/Ñ)
- Punctuation and formatting marks

## Files

- **`train.py`** - Main training script
- **`evaluate.py`** - Model evaluation and validation script

---

## Training (`train.py`)

### Basic Usage

```bash
# Train with default settings (YOLOv8n, 100 epochs)
python scripts/train.py \
  --data config/data.yaml \
  --model yolov8n \
  --epochs 100 \
  --batch 16

# Train with custom weights
python scripts/train.py \
  --data config/data.yaml \
  --weights path/to/weights.pt \
  --epochs 200 \
  --batch 32
```

### Key Arguments

#### Model Selection

- `--model` - Model architecture (default: `yolov8n`)
  - `yolov8n` - Nano (fastest, least accurate)
  - `yolov8s` - Small
  - `yolov8m` - Medium
  - `yolov8l` - Large
  - `yolov8x` - Extra Large (slowest, most accurate)
- `--weights` - Path to pretrained weights (optional)

#### Training Parameters

- `--data` - Path to data.yaml (default: `config/data.yaml`)
- `--epochs` - Number of training epochs (default: 100)
- `--batch` - Batch size (default: 16)
- `--imgsz` - Input image size (default: 640)
- `--device` - Device to use (`cuda`, `cpu`, `mps`)

#### Hyperparameters

- `--lr0` - Initial learning rate (default: 0.01)
- `--patience` - Early stopping patience (default: 50)
- `--save-period` - Save checkpoint every N epochs (default: -1, disabled)

#### Data Augmentation

- `--augment` - Enable augmentation
- `--mosaic` - Mosaic augmentation probability (default: 1.0)
- `--mixup` - Mixup augmentation probability (default: 0.0)
- `--copy-paste` - Copy-paste augmentation (default: 0.0)
- `--degrees` - Image rotation degrees (default: 0.0)
- `--translate` - Image translation fraction (default: 0.1)
- `--scale` - Image scale gain (default: 0.5)
- `--flipud` - Vertical flip probability (default: 0.0)
- `--fliplr` - Horizontal flip probability (default: 0.5)
- `--hsv-h` - HSV Hue augmentation (default: 0.015)
- `--hsv-s` - HSV Saturation augmentation (default: 0.7)
- `--hsv-v` - HSV Value augmentation (default: 0.4)

#### Output Options

- `--name` - Experiment name (auto-generated if not provided)
- `--project` - Project directory (default: model output directory)
- `--resume` - Resume training from last checkpoint

### Example Training Configurations

#### Quick Training (Testing)

```bash
python scripts/train.py \
  --model yolov8n \
  --epochs 50 \
  --batch 32 \
  --imgsz 640 \
  --name quick_test
```

#### Production Training (Recommended)

```bash
python scripts/train.py \
  --model yolov8m \
  --epochs 200 \
  --batch 16 \
  --imgsz 640 \
  --lr0 0.01 \
  --patience 50 \
  --augment \
  --mosaic 1.0 \
  --fliplr 0.5 \
  --name production_model
```

#### Fine-tuning Existing Model

```bash
python scripts/train.py \
  --weights runs/train/previous_model/weights/best.pt \
  --epochs 100 \
  --batch 16 \
  --lr0 0.001 \
  --name finetuned_model
```

### Output Structure

After training, you'll find:

```
runs/train/your_experiment_name/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.png          # Training metrics graphs
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png        # F1 score curve
├── P_curve.png         # Precision curve
├── R_curve.png         # Recall curve
└── args.yaml           # Training arguments
```

---

## Evaluation (`evaluate.py`)

### Basic Usage

```bash
# Evaluate on validation set
python scripts/evaluate.py \
  --weights runs/train/your_model/weights/best.pt \
  --data config/data.yaml

# Evaluate on test set
python scripts/evaluate.py \
  --weights runs/train/your_model/weights/best.pt \
  --data config/data.yaml \
  --split test
```

### Key Arguments

- `--weights` - Path to trained model weights (required)
- `--data` - Path to data.yaml file
- `--split` - Dataset split to evaluate (`train`, `val`, `test`)
- `--batch` - Batch size (default: 16)
- `--imgsz` - Image size (default: 640)
- `--conf` - Confidence threshold (default: 0.001)
- `--iou` - IoU threshold for NMS (default: 0.6)
- `--max-det` - Maximum detections per image (default: 300)
- `--save-json` - Save results to JSON
- `--save-hybrid` - Save hybrid labels

### Understanding Metrics

The evaluation script provides:

1. **Overall Metrics**

   - **Precision** - Accuracy of positive predictions
   - **Recall** - Ability to find all positive samples
   - **mAP50** - Mean Average Precision at IoU threshold 0.5
   - **mAP50-95** - Mean Average Precision averaged over IoU thresholds 0.5-0.95

2. **Per-Class Metrics**
   - Individual mAP scores for each Braille character
   - Helps identify which characters the model struggles with

### Example Evaluation Commands

```bash
# Standard evaluation
python scripts/evaluate.py \
  --weights runs/train/production_model/weights/best.pt \
  --data config/data.yaml \
  --split val

# Detailed evaluation with JSON export
python scripts/evaluate.py \
  --weights runs/train/production_model/weights/best.pt \
  --data config/data.yaml \
  --split test \
  --save-json \
  --conf 0.25 \
  --iou 0.45
```

---

## Data Configuration

Your `config/data.yaml` should look like:

```yaml
path: /path/to/braille_dataset
train: train/images
val: val/images
test: test/images

nc: 37 # Number of classes
names:
  - a
  - b
  - c
  # ... all Braille characters
  - capital
  - number
  - enye
```

---

## Best Practices

### 1. **Data Preparation**

- Ensure balanced dataset across all Braille characters
- Include diverse lighting conditions and backgrounds
- Annotate carefully, especially for similar-looking characters

### 2. **Model Selection**

- Start with `yolov8n` for quick iterations
- Use `yolov8m` or `yolov8l` for production
- Consider compute resources and inference speed requirements

### 3. **Training Strategy**

- Begin with default hyperparameters
- Monitor validation loss for overfitting
- Use early stopping (`--patience`) to prevent overtraining
- Save checkpoints periodically for long training runs

### 4. **Augmentation**

- Enable augmentation for better generalization
- Be careful with rotation (Braille orientation matters)
- Use horizontal flips cautiously (may confuse similar characters)

### 5. **Evaluation**

- Always evaluate on a held-out test set
- Check per-class metrics to identify problem characters
- Test on real-world images, not just validation set

---

## Troubleshooting

### Low mAP Scores

- Increase training epochs
- Try a larger model architecture
- Check data quality and annotations
- Adjust confidence threshold during evaluation

### Overfitting (train accuracy >> val accuracy)

- Reduce model size
- Increase augmentation
- Add more training data
- Use early stopping

### Slow Training

- Reduce batch size
- Use smaller model (yolov8n/yolov8s)
- Enable mixed precision training (`--half`)
- Use GPU if available

### Out of Memory Errors

- Reduce `--batch` size
- Reduce `--imgsz`
- Use smaller model architecture
- Close other applications

---

## Next Steps

After training and evaluation:

1. Review metrics and confusion matrix
2. Identify problematic character classes
3. Collect more data for underperforming classes
4. Fine-tune or retrain as needed
5. Proceed to prediction phase (see `USAGE.md`)

---
