# 🔤 Filipino Braille Detection & Text Conversion System

A robust, AI-powered system for detecting and converting Filipino Grade 1 Braille to readable text using YOLOv8 object detection and intelligent error correction.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00C9FF)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ Features

### 🎯 Core Capabilities

- **Real-time Braille Detection** - Detects individual Braille characters using YOLOv8
- **Bilingual Support** - Filipino/Tagalog and English text conversion
- **Smart Text Conversion** - Organizes detections into lines, words, and sentences
- **Special Characters** - Full support for enye (Ñ), numbers, and punctuation
- **Multiple Input Sources** - Images, videos, directories, and live camera feeds

### 🧠 Intelligent Error Correction

- **Bilingual Spell Checking** - Filipino + English dictionary-based corrections
- **AI-Powered Context Correction** - LLM-based intelligent text fixing
- **Gap Detection** - Identifies potentially missing characters
- **Quality Scoring** - Automatic assessment of detection quality
- **Detailed Reporting** - Comprehensive correction and confidence reports

### 🚀 Production-Ready

- **Batch Processing** - Process multiple images efficiently
- **Image Standardization** - Automatic resolution normalization
- **Flexible Output** - Text files, annotated images, detailed reports
- **Parameter Tuning** - Adjustable thresholds and spacing parameters
- **Camera Support** - Real-time detection with live preview

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [System Architecture](#-system-architecture)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Acknowledgemenets](#-acknowledgments)
- [Roadmap](#️-roadmap)

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, optional)
- Webcam (for camera mode, optional)

### Step 1: Clone Repository

```bash
git clone https://github.com/Shibarashii/filipino-braille-detection.git
cd filipino-braille-detection
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Core Dependencies:**

```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
pyspellchecker>=0.7.0
requests>=2.31.0
```

### Step 3: Download Pre-trained Model (Optional)

```bash
# Download YOLOv8 base model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 4: Set Up LLM API (Optional)

For AI-powered text correction:

```bash
# Get free API key from https://console.groq.com
echo "GROQ_API=your_api_key_here" > .env
```

**Alternative: Use Local LLM (Ollama)**

```bash
# Install Ollama (100% free, private)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama serve
```

---

## 🚀 Quick Start

### 1. Predict on Image

```bash
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source path/to/braille_image.jpg \
  --save \
  --save-text-file
```

### 2. Live Camera Detection

```bash
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source 0 \
  --show-text
```

### 3. Batch Process Directory

```bash
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source path/to/images/ \
  --enable-spellcheck \
  --save-text-file
```

### 4. High-Quality with AI Correction

```bash
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source braille.jpg \
  --enable-spellcheck \
  --enable-llm \
  --target-language both \
  --save-report
```

---

## 💡 Usage Examples

### Example 1: Basic Detection

```bash
# Detect Braille and convert to text
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --conf 0.25
```

**Output:**

- Annotated image with bounding boxes
- Converted text printed to console

### Example 2: Filipino Text with Corrections

```bash
# Enable Filipino spell checking
python scripts/predict.py \
  --weights model.pt \
  --source filipino_braille.jpg \
  --enable-spellcheck \
  --target-language tl \
  --save-text-file
```

**Output:**

- `filipino_braille_text.txt` with corrected text
- Console shows corrections made

### Example 3: Mixed Language with LLM

```bash
# Process mixed Filipino-English with AI correction
python scripts/predict.py \
  --weights model.pt \
  --source mixed_text.jpg \
  --enable-spellcheck \
  --enable-llm \
  --llm-api groq \
  --target-language both \
  --save-report
```

**Output:**

- Text file with AI-corrected content
- Detailed report showing all corrections
- Quality score and confidence metrics

### Example 4: Real-Time Camera with Text Overlay

```bash
# Live detection with on-screen text
python scripts/predict.py \
  --weights model.pt \
  --source 0 \
  --show-text \
  --show-fps \
  --enable-spellcheck
```

**Controls:**

- `q` - Quit
- `s` - Save snapshot
- `+/-` - Adjust confidence threshold

---

## 🏗️ System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Sources                            │
│  (Images, Videos, Directories, Camera Streams)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              YOLOv11 Braille Detector                       │
│  - Detects individual Braille characters                    │
│  - Bounding boxes + confidence scores                       │
│  - Real-time inference support                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Robust Braille Converter                           │
│  - Organizes detections into lines/words                    │
│  - Handles capital letters and numbers                      │
│  - Gap detection for missing characters                     │
│  - Quality scoring                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Error Correction Pipeline                         │
│                                                             │
│  ┌──────────────────┐      ┌────────────────────┐           │
│  │ Spell Checking   │ ───▶ │  LLM Correction    │           │
│  │ (Filipino + EN)  │      │  (Context-aware)   │           │
│  └──────────────────┘      └────────────────────┘           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Generation                        │
│  - Corrected text files                                     │
│  - Annotated images                                         │
│  - Detailed correction reports                              │
│  - Quality metrics                                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **`predict.py`** - Main prediction script

   - Handles all input sources
   - Coordinates detection and conversion
   - Manages output generation

2. **`braille_converter.py`** - Text conversion engine

   - Spatial organization of detections
   - Bilingual spell checking
   - Quality assessment
   - Parameter optimization suggestions

3. **`llm.py`** - AI text corrector

   - Context-aware error correction
   - Multi-language support
   - Multiple LLM provider support

4. **`train.py`** - Model training

   - YOLOv8 fine-tuning
   - Augmentation pipeline
   - Hyperparameter optimization

5. **`evaluate.py`** - Model evaluation
   - Performance metrics
   - Per-class analysis
   - Confusion matrices

---

## ⚙️ Configuration

### Detection Parameters

```bash
--conf 0.25              # Confidence threshold (0.0-1.0)
--iou 0.45               # IoU threshold for NMS
--imgsz 640              # Input image size
--max-det 300            # Maximum detections per image
```

### Conversion Parameters

```bash
--line-height 50         # Vertical distance for line grouping (pixels)
--word-gap 80            # Horizontal gap for word separation (pixels)
--char-gap 30            # Expected character spacing (pixels)
--min-confidence 0.10    # Minimum detection confidence
```

### Correction Parameters

```bash
--enable-spellcheck      # Enable dictionary-based corrections
--enable-llm             # Enable AI-powered corrections
--llm-api groq           # LLM provider (groq/ollama/together/huggingface)
--target-language en     # Language (en/tl/both)
```

### Output Parameters

```bash
--save                   # Save annotated images
--save-text-file         # Save converted text
--save-report            # Save detailed report
--show-text              # Display text on image
--view-img               # Display results window
```

---

## 📚 Documentation

Detailed documentation is available in the `/scripts/` directory:

- **[TRAINING.md](scripts/TRAINING.md)** - Complete guide to training and evaluating models

  - Model selection and architecture
  - Training strategies and hyperparameters
  - Data augmentation techniques
  - Performance evaluation
  - Troubleshooting training issues

- **[USAGE.md](scripts/USAGE.md)** - Comprehensive usage guide
  - Prediction on images, videos, and camera
  - Text conversion and error correction
  - LLM setup and configuration
  - Advanced features and parameters
  - Complete examples and workflows

---

### Getting Help

1. **Check Documentation**

   - [TRAINING.md](scripts/TRAINING.md) for training issues
   - [USAGE.md](scripts/USAGE.md) for prediction issues

2. **Enable Verbose Output**

   ```bash
   --verbose
   ```

3. **Check Logs**

   - Training logs: `runs/train/experiment_name/`
   - Prediction reports: Review detailed reports with `--save-report`

4. **Review Metrics**
   ```bash
   # Get parameter suggestions
   --save-report
   # Check report file for recommendations
   ```

---

## 🎯 Project Structure

```
braille-detection/

├── config
│   ├── data.yaml
│   └── model_config.py
├── data
│   └── grade1_dataset_updated
│       ├── test
│       │   ├── images
│       │   └── labels
│       ├── train
│       │   ├── images
│       │   └── labels
│       ├── valid
│       │   ├── images
│       │   └── labels
│       └── data.yaml
├── models
│   └── yolo_detector.py
├── outputs
│   ├── logs
│   ├── models
│   └── predictions
├── runs
│   └── detect
│       └── val
├── scripts
│   ├── analyze_dataset.py
│   ├── batch_inference.py
│   ├── braille_converter.py
│   ├── download_models.py
│   ├── evaluate.py
│   ├── export.py
│   ├── llm.py
│   ├── predict.py
│   └── train.py
├── utils
│   ├── logger.py
│   └── visualization.py
├── notebook.ipynb
├── QUICKSTART.md
├── README.md
├── TRAINING.md
└── USAGE.md
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **Dataset Expansion**

   - Add more Braille samples
   - Improve character diversity
   - Add different Braille grades

2. **Model Improvements**

   - Optimize hyperparameters
   - Test different architectures
   - Improve detection accuracy

3. **Feature Additions**

   - Support for Braille Grade 2
   - Multi-page document processing
   - Mobile app integration
   - Web interface

4. **Documentation**
   - Translate to other languages
   - Add more examples
   - Create video tutorials

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/braille-detection.git
cd braille-detection

# Create development branch
git checkout -b feature/your-feature-name

# Make changes and test
python scripts/predict.py --weights model.pt --source test.jpg

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### Pull Request Guidelines

1. Ensure code follows existing style
2. Add tests for new features
3. Update documentation
4. Include example usage
5. Describe changes clearly

---

## 📜 License

See `LICENSE.txt`

---

## 🙏 Acknowledgments

- **Sir Reyes** - Shout out to you sir

---

## 🗺️ Roadmap

### Current Version (v1.0)

- ✅ YOLOv8-based Braille detection
- ✅ Bilingual text conversion (Filipino + English)
- ✅ Spell checking
- ✅ LLM-based correction
- ✅ Camera support
- ✅ Batch processing

### Planned Features (v2.0)

- 🔄 Braille Grade 2 support
- 🔄 Multi-page document processing
- 🔄 Mobile app (iOS/Android)
- 🔄 Web interface
- 🔄 REST API
- 🔄 Cloud deployment support

### Future Vision (v3.0)

- 📋 Real-time translation
- 📋 Braille writing assistance
- 📋 Educational tools
- 📋 Accessibility features
- 📋 Multi-language support (beyond Filipino)

---

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/Shibarashii/filipino-braille-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/Shibarashii/filipino-braille-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/Shibarashii/filipino-braille-detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Shibarashii/filipino-braille-detection)

---

<div align="center">

**Made with ❤️ for the visually impaired community**

[⬆ Back to Top](#-filipino-braille-detection--text-conversion-system)

</div>
