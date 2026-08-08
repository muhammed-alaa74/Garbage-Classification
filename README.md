# MLJ — Smart Garbage Classification

An AI system that identifies what type of waste is in a photo and tells you how to dispose of it correctly. Built with transfer learning on 10 waste categories and deployed as a live Streamlit app with Arabic-language recycling guidance.

**[Live App →](https://garbage-classification-2m.streamlit.app/)**

<p align="center">
  <img src="assets/demo_upload_result.png" alt="MLJ app — upload and prediction result" width="80%"/>
</p>

---

## The Problem

Most people don't know which bin a given item belongs to, and incorrect sorting is one of the biggest reasons recyclable material ends up in landfills. MLJ solves this at the point of decision: take a photo of an item, and the app tells you exactly what it is and how to dispose of it — no guessing.

## What It Does

1. You upload a photo of a single waste item.
2. A trained CNN classifies it into one of 10 categories and shows a confidence score, along with the top-3 most likely classes.
3. The app returns a short, practical tip in Arabic on how to correctly dispose of, recycle, or reuse that specific item.

<p align="center">
  <img src="assets/demo_top3_and_tips.png" alt="MLJ app — top-3 predictions and disposal guidance" width="80%"/>
</p>

---

## Table of Contents

- [The Problem](#the-problem)
- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Running Locally](#running-locally)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## How It Works

Two CNN architectures were trained and compared, both using transfer learning from ImageNet weights:

| Model | Approach | Test Accuracy |
|---|---|---|
| **MobileNetV2** | Frozen base → head training → fine-tune last 30 layers | 89.3% |
| **EfficientNetB0** | Frozen base → head training → fine-tune last 30 layers | **91.5%** |

The deployed app lets you pick either model at inference time.

---

## Dataset

- **Source:** [Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (Kaggle), `standardized_256` variant.
- **Classes (10):** battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash.
- **Size:** ~11,259 images, imbalanced across classes (453–1,892 images per class).
- **Split:** 80% train / 10% validation / 10% test via `split-folders`, fixed seed for reproducibility.

<p align="center">
  <img src="assets/class_distribution.png" alt="Class distribution" width="65%"/>
</p>

Because the dataset is imbalanced (`clothes` has roughly 4x more images than `trash`), class weights (`sklearn.utils.class_weight.compute_class_weight`, `balanced`) were passed to `model.fit()` so the model doesn't just default to the majority classes. All images were also verified with `PIL.Image.verify()` to screen out corrupted files before training.

Full EDA — per-class counts, sample grids, dimension checks, file-extension audit — is in the training notebook.

---

## Model Training

**Preprocessing and augmentation** (`ImageDataGenerator`):
- Rescale (`1./255` for MobileNetV2, `preprocess_input` for EfficientNetB0)
- Rotation ±20°, width/height shift ±10%, horizontal flip, zoom ±15%, brightness jitter 0.8–1.2
- Input size 224×224, batch size 32

**Two-stage transfer learning per architecture:**
1. **Head training** — base frozen, custom head (`GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.3) → Dense(10, softmax)`) trained up to 15 epochs, `Adam(lr=1e-3)`.
2. **Fine-tuning** — last 30 layers of the base unfrozen, trained up to 10 more epochs, `Adam(lr=1e-5)`.

**Callbacks:** `EarlyStopping` (patience 5, restores best weights), `ReduceLROnPlateau` (factor 0.5, patience 3), `ModelCheckpoint` (best validation accuracy).

The full pipeline — data loading, EDA, splitting, augmentation, training, fine-tuning, and evaluation — is in [`notebooks/garbage-classification.ipynb`](notebooks/garbage-classification.ipynb). It was built and run on Kaggle; update `DATASET_PATH` at the top to reproduce with your own copy of the dataset.

---

## Results

<p align="center">
  <img src="assets/training_comparison.png" alt="Training accuracy and loss comparison" width="85%"/>
</p>

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion matrix — MobileNetV2" width="65%"/>
</p>

Most confusion happens between visually similar categories — glass vs. metal/plastic, and paper vs. cardboard — which is expected given overlapping shapes, colors, and packaging materials.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/muhammed-alaa74/Garbage-Classification.git
cd Garbage-Classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add the trained model(s) — see note below
# Place mobilenetv2_garbage_classifier.keras and/or
# efficientnetb0_garbage_classifier.keras inside models/

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> **Model files are not in this repository** (large binaries, excluded via `.gitignore`). Train the model using the notebook and drop the resulting `.keras` file(s) into `models/`, or host them externally (Hugging Face Hub, Google Drive, Git LFS, or a GitHub Release) and download them on app startup.

---

## Project Structure

```
garbage-classification/
├── app.py                      # Streamlit inference app
├── requirements.txt
├── LICENSE
├── README.md
├── notebooks/
│   └── garbage-classification.ipynb   # training & EDA notebook
├── assets/                     # images used in this README / app
└── models/                     # trained .keras model(s) go here (gitignored)
    ├── mobilenetv2_garbage_classifier.keras
    └── efficientnetb0_garbage_classifier.keras
```

---

## Tech Stack

- **TensorFlow / Keras** — model architecture, transfer learning, training
- **MobileNetV2 / EfficientNetB0** — ImageNet-pretrained backbones
- **scikit-learn** — classification report, confusion matrix, class weighting
- **split-folders** — reproducible train/val/test splitting
- **Streamlit** — interactive web app for inference
- **Pandas / Matplotlib / Seaborn** — EDA and result visualization

---

## License

Released under the [MIT License](LICENSE).