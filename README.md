# 🧪 Cells Instance Segmentation — Mask R-CNN Starter

Instance segmentation baseline for microscopy **cell masks** using **PyTorch Mask R-CNN** with a **ConvNeXt-Tiny + FPN** backbone. The notebook covers **end-to-end** training, validation, inference, and submission generation with **RLE** encoding.

<img width="1265" height="356" alt="image" src="https://github.com/user-attachments/assets/c1faa78a-be60-445d-8a2b-7b1f9c9cbf78" />

---

## 📘 What’s Inside

- **Reproducible setup**
  - Global seed fixing (NumPy, Python, PyTorch, cuDNN)
  - Centralized config for paths, device, image size, and hyperparameters

- **Data & Augmentations**
  - Albumentations pipelines (train & test)
  - Utilities for **RLE decode/encode** and **overlap removal**
  - Custom `CellDataset` (masks + boxes) and `CellTestDataset` (inference)
  - `DataLoader`s with custom `collate_fn`

- **Model**
  - Mask R-CNN with **ConvNeXt-Tiny** (via `timm`) + **Feature Pyramid Network**
  - Two classes: **background** + **cell**
  - Tunable `box_detections_per_img`

- **Training & Validation**
  - Mixed precision (AMP) with `GradScaler`
  - Optimizer: **AdamW**; Scheduler: **ReduceLROnPlateau**
  - Loss logging (total & mask loss)
  - **Best checkpoint** saved by lowest **validation loss**

- **Inference & Submission**
  - Batched inference on test images
  - Confidence filtering, **mask binarization**, and **overlap removal**
  - **RLE encoding** → `submission.csv`
  - Helper to visualize predicted masks

---

## 🗺️ Paths & Configuration

- **Input**
  - `TRAIN_CSV` — annotations (`id`, `annotation` in RLE)
  - `TRAIN_PATH` — training images (`.png`)
  - `TEST_PATH` — test images (`.png`)

- **Core Params**
  - `WIDTH=704`, `HEIGHT=520` (match dataset resolution for segmentation quality)
  - `PCT_IMAGES_VALIDATION=0.1`
  - `BATCH_SIZE=1` (Mask R-CNN is memory heavy; tune per GPU)
  - `NUM_EPOCHS=20`, `LEARNING_RATE=1e-4`, `WEIGHT_DECAY=5e-4`
  - Inference: `BOXES_CONF`, `MASK_THRESHOLD`

> 🔐 Seeds fixed via `fix_all_seeds(2025)` for best-effort reproducibility.

---

## 🧱 Data Splits

- **Group-aware split by image `id`**
  - Ensures **no leakage** of masks from the same image between train/val
  - Reported example:
    - Train: **381 images / 48,918 annotations**
    - Val: **43 images / 4,147 annotations**

---

## 🧩 Datasets

### `CellDataset` (Train/Val)
- Inputs:
  - image (`H×W×3`), decoded **instance masks** (from RLE)
- Outputs:
  - `img` (tensor)
  - `target` dict: `boxes`, `labels`, `masks`, `image_id`, `area`, `iscrowd`
- Handles:
  - Empty instances
  - Invalid boxes (filtered)
  - Augmentations with synchronized `bboxes` / `masks`

### `CellTestDataset` (Inference)
- Iterates over **test images** and returns:
  - `image` (tensor), `image_id` (str)

---

## 🧯 Augmentations

**Train** (probabilistic):
- Resize → Flips/Rotations → Shift/Scale/Rotate → Noise/Blur → Color Jitter
- Normalize + `ToTensorV2`

**Test**:
- Resize → Normalize + `ToTensorV2`

> All augmentations respect masks and bounding boxes.

---

## 🧠 Model

- **Backbone:** `ConvNeXt-Tiny` (from `timm`) → **FPN**
- **Detector:** `torchvision.models.detection.MaskRCNN`
- **Heads:**
  - Box head (classification + bbox regression)
  - Mask head (per-instance masks)
- **Classes:** 2 (bg, cell)

---

## 🏋️ Training Loop

- Mixed precision (`autocast`, `GradScaler`)
- Gradient clipping (`max_norm=1.0`)
- NaN/Inf loss guard (skip batch)
- **ReduceLROnPlateau** on **val loss**
- **Best model** saved to `best_model.bin` (lowest val loss)

**Logged per epoch:**
- Train: total loss, mask loss
- Val: total loss, mask loss
- Elapsed time

---

## 🔎 Inference

1. Load **best checkpoint**.
2. Forward pass on test images.
3. For each detection:
   - Filter by **`scores` ≥ `BOXES_CONF`**
   - Binarize mask with **`MASK_THRESHOLD`**
   - **Remove overlaps** with previously accepted masks
   - **RLE encode** and append to submission
4. If no valid masks → `annotation = "-1"`

**Output:** `submission.csv` with columns `idx`, `id`, `annotation`.

---

## 🖼️ Visualization


<img width="989" height="1346" alt="mask" src="https://github.com/user-attachments/assets/169c3402-bc2d-419d-b561-c55fc1758729" />

---

## ⚙️ Tips & Troubleshooting

- **OOM?** Lower `BATCH_SIZE`, or reduce `WIDTH/HEIGHT`, or freeze backbone initially.
- **Poor masks?**
  - Adjust `MASK_THRESHOLD` (0.3–0.7)
  - Increase `NUM_EPOCHS`
  - Stronger/softer augmentation
  - Try higher `BOXES_CONF` (e.g., 0.3–0.6)
- **No detections?**
  - Lower `BOXES_CONF`
  - Check image normalization stats
  - Verify masks decode to correct size (`HEIGHT×WIDTH`)
- **Noisy overlaps?** Keep **overlap removal** enabled (priority to higher-score masks).

---

## ✅ Deliverables

- **Training logs** (train/val loss per epoch)
- **`best_model.bin`** (lowest val loss)
- **`submission.csv`** (RLE per instance per image)
- Optional: qualitative figures (augmented samples, predictions)

---

## 📌 Summary

This starter sets up a **strong, extensible baseline** for cell instance segmentation:
- Reliable data pipeline with **Albumentations**, **RLE**, and **overlap removal**
- Competitive **Mask R-CNN** with a modern **ConvNeXt + FPN** backbone
- **AMP training**, robust logging, and **best-checkpoint** selection
- Turn-key **inference → submission** flow

Use it as a foundation; iterate on **backbones, thresholds, augmentations, and training schedules** to push performance on your dataset.

