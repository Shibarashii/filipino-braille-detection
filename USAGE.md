# Braille Detection and Text Conversion Guide

This guide covers using the trained model to detect Braille and convert it to readable text with intelligent error correction.

## Overview

The prediction system consists of three main components:

1. **`predict.py`** - Main prediction script with image/video/camera support
2. **`braille_converter.py`** - Robust Braille-to-text converter with spell checking
3. **`llm.py`** - AI-powered context-aware text correction

---

## Quick Start

### Basic Image Prediction

```bash
# Predict on a single image
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source path/to/braille_image.jpg

# Predict on all images in a directory
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source path/to/images/
```

### Live Camera Detection

```bash
# Use default camera (camera 0)
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source 0

# Use specific camera
python scripts/predict.py \
  --weights runs/train/your_model/weights/best.pt \
  --source 1
```

---

## Core Features

### 1. Braille Detection

- Detects individual Braille characters using YOLO
- Supports Filipino Grade 1 Braille (letters, numbers, enye/Ñ)
- Real-time detection on camera/video

### 2. Text Conversion

- Organizes detections into lines and words
- Handles capital letters and numbers
- Preserves Filipino special characters (Ñ)

### 3. Error Correction (Optional)

#### Bilingual Spell Checking

- Corrects spelling errors in English and Filipino/Tagalog
- Suggests alternatives for misspelled words
- Maintains original capitalization

#### AI-Powered Context Correction

- Uses LLM (Large Language Model) to fix missing/wrong letters
- Understands context to make intelligent corrections
- Supports English, Filipino, or mixed language text

---

## Prediction Script (`predict.py`)

### Detection Parameters

```bash
python scripts/predict.py \
  --weights runs/train/model/weights/best.pt \
  --source image.jpg \
  --conf 0.25 \           # Confidence threshold
  --iou 0.45 \            # IoU threshold for NMS
  --imgsz 640 \           # Input image size
  --device cuda \         # Use GPU (cuda/cpu/mps)
  --max-det 300           # Max detections per image
```

### Text Conversion Parameters

```bash
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --line-height 50 \      # Max vertical distance for same line (pixels)
  --word-gap 80 \         # Min horizontal gap for word spacing (pixels)
  --char-gap 30 \         # Expected character spacing (pixels)
  --min-confidence 0.10   # Minimum detection confidence
```

### Spell Checking (English & Filipino)

```bash
# Enable bilingual spell checking
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --enable-spellcheck

# Disable spell checking
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --disable-spellcheck
```

### AI-Powered LLM Correction

```bash
# Enable LLM correction for English
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --enable-llm \
  --llm-api groq \
  --llm-key your_groq_api_key \
  --target-language en

# For Filipino text
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --enable-llm \
  --target-language tl

# For mixed Filipino + English
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --enable-llm \
  --target-language both
```

**Target Language Options:**

- `en` - English only
- `tl` - Filipino/Tagalog only
- `both` - Mixed Filipino and English

### Image Standardization

Resize images to consistent resolution for better results:

```bash
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --standardize-size \
  --output-width 1280 \
  --output-height 720 \
  --keep-aspect-ratio
```

### Output Options

```bash
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --save \                # Save annotated images
  --save-txt \            # Save detection labels
  --save-text-file \      # Save converted text
  --save-report \         # Save detailed correction report
  --show-text \           # Display text on image
  --view-img              # Display results
```

### Camera Mode Controls

When using camera input:

- **`q`** or **`ESC`** - Quit
- **`s`** - Save current frame and text
- **`+`** or **`=`** - Increase confidence threshold
- **`-`** or **`_`** - Decrease confidence threshold

---

## Braille Converter (`braille_converter.py`)

### Basic Usage

```python
from scripts.braille_converter import RobustBrailleConverter

# Initialize converter
converter = RobustBrailleConverter(
    line_height_threshold=50,    # Vertical grouping
    word_gap_threshold=80,        # Word spacing
    char_gap_threshold=30,        # Character spacing
    min_confidence=0.10,          # Min detection confidence
    enable_spellcheck=True,       # Enable spell checking
    enable_gap_detection=True,    # Detect missing characters
    bilingual=True,               # Filipino + English
    enable_llm_correction=True,   # Enable AI correction
    llm_api='groq',               # LLM provider
    llm_api_key='your_key',       # API key
    target_language='en'          # Target language
)

# Convert YOLO results to text
text = converter.convert_results_to_text(results)
print(text)

# Get detailed report
report = converter.get_detection_report(results)
print(f"Corrections: {report['corrections_made']}")
print(f"Quality Score: {report['quality_score']:.2f}")
print(f"Final Text: {report['final_text']}")
```

### Features

#### 1. Line and Word Organization

- Groups detections into lines based on vertical position
- Separates words based on horizontal gaps
- Maintains proper text structure

#### 2. Gap Detection

- Identifies unusually large gaps between characters
- Warns about potentially missing characters
- Helps improve detection quality

#### 3. Bilingual Spell Checking

- Built-in Filipino and English dictionaries
- Corrects common OCR errors
- Suggests alternatives for unknown words

```python
# Get spelling suggestions
suggestions = converter.get_correction_suggestions("kumian")
# Returns: {'tl': ['kumain', 'kumian', ...], 'en': [...]}

# Add custom Filipino words
converter.add_custom_filipino_words([
    'Pangatlong', 'Baitang', 'Modyul'
])
```

#### 4. Quality Scoring

- Calculates overall detection quality
- Considers confidence, corrections, and completeness
- Provides actionable feedback

#### 5. Parameter Optimization

```python
# Get recommendations for better results
suggestions = converter.suggest_parameter_adjustments(results)
for key, msg in suggestions.items():
    print(msg)
```

---

## LLM Text Corrector (`llm.py`)

### Supported LLM Providers

1. **Groq** (Recommended) - Fast and free
2. **Ollama** - 100% local and private
3. **Together AI** - Free tier available
4. **HuggingFace** - Free inference API

### Setting Up LLM Correction

#### Option 1: Groq (Fastest, Free)

```bash
# 1. Get API key from https://console.groq.com
# 2. Create .env file:
echo "GROQ_API=your_groq_api_key_here" > .env

# 3. Use in prediction
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --enable-llm \
  --llm-api groq \
  --target-language en
```

#### Option 2: Ollama (Local, Private)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Download model
ollama pull llama3.2

# 3. Start Ollama server
ollama serve

# 4. Use in prediction
python scripts/predict.py \
  --weights model.pt \
  --source image.jpg \
  --enable-llm \
  --llm-api ollama \
  --target-language en
```

### Programmatic Usage

```python
from scripts.llm import LLMTextCorrector

# Initialize corrector
corrector = LLMTextCorrector(
    api_choice='groq',
    api_key='your_api_key'
)

# Correct English text
result = corrector.correct_text(
    "brille nable blind peple to rad",
    language='en'
)
print(result['corrected_text'])
# Output: "braille enable blind people to read"

# Correct Filipino text
result = corrector.correct_text(
    "kumian ako ng tinapy",
    language='tl'
)
print(result['corrected_text'])
# Output: "kumain ako ng tinapay"

# Correct mixed language text
result = corrector.correct_text(
    "Ang brille ay tumutulog sa blind na tao",
    language='both'
)
```

### Language Support

- **`language='en'`** - English text correction
- **`language='tl'`** - Filipino/Tagalog text correction
- **`language='both'`** - Mixed Filipino and English

### API Configuration

```python
# Get free API setup instructions
from scripts.llm import get_free_api_instructions
get_free_api_instructions()
```

---

## Complete Examples

### Example 1: Basic Prediction

```bash
python scripts/predict.py \
  --weights runs/train/braille_model/weights/best.pt \
  --source sample_images/braille_page.jpg \
  --conf 0.25 \
  --save \
  --save-text-file
```

**Output:**

- Annotated image with bounding boxes
- `braille_page_text.txt` with converted text

### Example 2: High-Quality with All Corrections

```bash
python scripts/predict.py \
  --weights runs/train/braille_model/weights/best.pt \
  --source braille_document.jpg \
  --enable-spellcheck \
  --enable-llm \
  --llm-api groq \
  --llm-key $GROQ_API \
  --target-language both \
  --save \
  --save-text-file \
  --save-report \
  --line-height 50 \
  --word-gap 80
```

**Output:**

- Annotated image
- `braille_document_text.txt` - Final corrected text
- `braille_document_report.txt` - Detailed correction report

### Example 3: Real-Time Camera Detection

```bash
python scripts/predict.py \
  --weights runs/train/braille_model/weights/best.pt \
  --source 0 \
  --enable-spellcheck \
  --show-text \
  --show-fps \
  --conf 0.3
```

**Features:**

- Live Braille detection
- Real-time text conversion
- On-screen text display
- FPS counter

### Example 4: Batch Processing Directory

```bash
python scripts/predict.py \
  --weights runs/train/braille_model/weights/best.pt \
  --source braille_images/ \
  --enable-spellcheck \
  --enable-llm \
  --target-language tl \
  --standardize-size \
  --save-text-file \
  --save-report
```

**Processes all images and generates:**

- Individual text files for each image
- Correction reports for each image
- Standardized resolution outputs

---

## Output Files

After prediction, you'll find:

```
runs/predict/predictions_TIMESTAMP/
├── image1.jpg              # Annotated image with boxes
├── image1_text.txt         # Converted text
├── image1_report.txt       # Detailed report
└── labels/
    └── image1.txt          # YOLO format labels
```

**Report Contents:**

- Total detections
- Number of lines
- Average confidence
- Quality score
- Spelling corrections made
- LLM corrections made
- Low confidence words
- Final converted text
- Raw text (before corrections)

---

## Troubleshooting

### Poor Detection Quality

1. Check image quality (resolution, lighting)
2. Adjust confidence threshold (`--conf`)
3. Enable image standardization (`--standardize-size`)
4. Retrain model with similar images

### Incorrect Text Conversion

1. Adjust spacing parameters:
   - `--line-height` for line grouping
   - `--word-gap` for word separation
   - `--char-gap` for character spacing
2. Enable gap detection (`--enable-gap-detection`)
3. Review low confidence words in report

### Spell Checking Issues

1. Add custom words to dictionary:
   ```python
   converter.add_custom_filipino_words(['word1', 'word2'])
   ```
2. Check language detection in corrections
3. Disable if causing problems (`--disable-spellcheck`)

### LLM Correction Not Working

1. **Groq**: Check API key is valid
2. **Ollama**: Ensure `ollama serve` is running
3. Check network connection
4. Review error messages in console
5. Try different LLM provider (`--llm-api`)

### Camera Performance Issues

1. Reduce resolution (`--camera-width`, `--camera-height`)
2. Increase confidence threshold (press `+`)
3. Use smaller model (yolov8n)
4. Disable text overlay (`--no-show-text`)

---

## Best Practices

### For Best Detection Results:

1. Use good lighting (avoid shadows)
2. Ensure Braille is in focus
3. Keep camera steady
4. Use appropriate distance from Braille
5. Standardize image resolution

### For Best Text Conversion:

1. Enable spell checking for cleaner output
2. Use LLM correction for context-aware fixes
3. Adjust spacing parameters based on your Braille source
4. Review correction reports to improve parameters
5. Choose correct target language

### For Production Use:

1. Save all outputs (`--save`, `--save-text-file`, `--save-report`)
2. Use appropriate confidence threshold (0.25-0.35)
3. Enable standardization for consistent results
4. Test on sample images before batch processing
5. Monitor quality scores in reports

---

## API Keys and Privacy

### Free API Options:

- **Groq**: 30 requests/min, 14,400/day (free tier)
- **Ollama**: Unlimited (runs locally)
- **Together AI**: Free tier available
- **HuggingFace**: Free inference API

### Privacy Considerations:

- **Ollama**: 100% private, runs on your computer
- **Cloud APIs**: Text is sent to external servers
- For sensitive content, use Ollama

### Setting up `.env` file:

```bash
# Create .env file in project root
echo "GROQ_API=your_api_key_here" > .env
```

---

## Performance Optimization

### For Speed:

```bash
--imgsz 320              # Smaller input size
--device cuda            # Use GPU
--half                   # FP16 inference
--disable-spellcheck     # Skip spell checking
--no-enable-llm          # Skip LLM correction
```

### For Accuracy:

```bash
--imgsz 640              # Larger input size
--conf 0.15              # Lower confidence threshold
--enable-spellcheck      # Enable spell checking
--enable-llm             # Enable LLM correction
--standardize-size       # Standardize resolution
```

---

## Next Steps

1. Test on sample images to verify setup
2. Adjust parameters based on your Braille source
3. Integrate into your application or workflow
4. Set up automated batch processing if needed
5. Monitor quality scores and iterate

For training new models or improving existing ones, see `TRAINING.md`.
