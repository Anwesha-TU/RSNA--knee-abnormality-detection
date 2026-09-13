# RSNA Knee MRI Abnormality Detection

A medical imaging project based on the **RSNA Knee Abnormality Detection** dataset, focused on building a DICOM-aware deep learning pipeline for **multi-label knee MRI abnormality classification**.

The project combines MRI preprocessing, metadata-aware series selection, 3D volume construction, a lightweight PyTorch 3D CNN, multilingual radiology-report weak supervision, and Grad-CAM-based model attention analysis.

---

## 📌 Project Overview

Knee MRI examinations contain multiple imaging series, anatomical planes, MRI sequences, and DICOM slices. Different abnormalities may be better visualized in different planes and sequence types.

This project develops a study-level pipeline that:

- Processes raw DICOM MRI series
- Orders MRI slices using spatial DICOM metadata
- Constructs 3D MRI volumes
- Selects representative MRI series based on anatomical plane and sequence characteristics
- Normalizes and resizes MRI volumes for deep learning
- Performs multi-label abnormality classification
- Handles severe class imbalance using weighted binary cross-entropy
- Explores multilingual radiology reports for weak supervision
- Uses masked learning for uncertain weak labels
- Applies 3D Grad-CAM for model attention analysis

---

## 🎯 Objective

Given a knee MRI study, predict the presence or absence of multiple abnormalities.

The task is formulated as a **multi-label classification problem**, since a single MRI study may contain multiple abnormalities simultaneously.

The 12 target abnormalities are:

- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Medial Osteoarthritis (OA)
- Lateral Osteoarthritis (OA)
- Patellofemoral Osteoarthritis (PF OA)
- Effusion
- Synovitis
- Baker's Cyst
- Contusion
- Fracture

---

## 📊 Dataset

The project uses the **RSNA Knee Abnormality Detection** dataset.

Current dataset exploration identified:

- **4,407 knee MRI studies**
- **24,371 imaging series**
- **58 studies with official abnormality labels**
- **4,349 studies with radiology reports but without official abnormality labels**
- **12 abnormality categories**

The imaging data is organized hierarchically:

```text
Study
│
├── Series 1
│   ├── DICOM Slice 1
│   ├── DICOM Slice 2
│   └── ...
│
├── Series 2
│   ├── DICOM Slice 1
│   └── ...
│
└── ...
```
The MRI series contain different anatomical planes and sequence characteristics, including:

- Sagittal
- Coronal
- Axial
- Proton Density (PD)
- T2-weighted
- Fluid-sensitive sequences
- Fat-suppressed sequences
- Other sequence types

🧠 Abnormality Classification

The model predicts 12 abnormalities simultaneously.

The problem is therefore represented as:
```text
MRI Study
    │
    ├── ACL
    ├── MCL
    ├── Medial Meniscus
    ├── Lateral Meniscus
    ├── Medial OA
    ├── Lateral OA
    ├── PF OA
    ├── Effusion
    ├── Synovitis
    ├── Baker's Cyst
    ├── Contusion
    └── Fracture
 ```   
Each output represents an independent abnormality probability.

🔬 Pipeline
1. Dataset Exploration

The project begins by analyzing the study and series structure of the dataset.

This includes:

- Counting studies and imaging series
- Examining study-to-series relationships
- Inspecting available anatomical planes
- Investigating MRI sequence characteristics
- Identifying officially labeled studies
- Examining class distributions
- Inspecting radiology reports

2. DICOM Processing

MRI images are stored as individual DICOM slices.

The preprocessing pipeline uses pydicom to read the DICOM files and inspect relevant metadata, including:

- Magnetic field strength
- Slice thickness
- Pixel spacing
- Image orientation
- Image position
- Instance number
- Series information

The DICOM metadata is used to determine the correct spatial ordering of slices.

When available, spatial information such as Image Position and Image Orientation is preferred over simple filename ordering.

3. 3D MRI Volume Construction

Ordered DICOM slices are stacked to form a 3D MRI volume.
```text
DICOM Slice 1
      ↓
DICOM Slice 2
      ↓
DICOM Slice 3
      ↓
      ...
      ↓
DICOM Slice N
      ↓
3D MRI Volume
```
This allows the model to process volumetric anatomical information rather than treating every MRI slice independently.

4. MRI Series Selection

A knee MRI study may contain several imaging series.

The project therefore uses metadata-aware series selection rather than randomly selecting a series.

Selection considers:

- Anatomical plane
- Fluid sensitivity
- Fat suppression
- MRI sequence characteristics
- Number of available slices

Representative series are selected for:

- Axial
- Coronal
- Sagittal

This provides multi-plane anatomical information for downstream classification.

5. Intensity Normalization

- MRI intensity values can vary between scans and scanners.

- The pipeline applies percentile-based intensity normalization to reduce the effect of extreme intensity values and improve consistency between volumes.

- The normalized volume is subsequently prepared for model input.

6. Volume Resizing

MRI volumes have different spatial dimensions.

For computational efficiency, the volumes are resized to a consistent representation:

[Channels, Depth, Height, Width]

[3, 32, 128, 128]

The three input channels correspond to representative:

- Channel 1 : Axial
- Channel 2 : Coronal
- Channel 3 : Sagittal

Missing planes are handled without discarding the complete study.

🧪 PyTorch Dataset

A custom PyTorch dataset was implemented to connect the processed MRI volumes with their corresponding study-level labels.

The dataset provides:
```text
MRI Study
    ↓
Axial Volume
Coronal Volume
Sagittal Volume
    ↓
Preprocessed Tensor
    ↓
3D CNN
```
The final input tensor has the shape:

[3, 32, 128, 128]

🤖 Baseline Deep Learning Model

A lightweight 3D Convolutional Neural Network (3D CNN) was implemented using PyTorch.

The model contains:

- 3D convolutional layers
- Batch normalization
- ReLU activations
- 3D max pooling
- Adaptive 3D global pooling
- Fully connected classification layers
- Dropout regularization

Architecture overview:
```text
Multi-plane MRI Tensor
        │
        ▼
   3D Convolution
        │
        ▼
 Batch Normalization
        │
        ▼
      ReLU
        │
        ▼
   3D Max Pooling
        │
        ▼
   3D Convolution
        │
        ▼
 Batch Normalization
        │
        ▼
      ReLU
        │
        ▼
   3D Max Pooling
        │
        ▼
   3D Convolution
        │
        ▼
 Adaptive 3D Pooling
        │
        ▼
 Fully Connected Layers
        │
        ▼
 12 Abnormality Outputs
```
The model is intentionally lightweight to allow experimentation in a CPU-only environment.

⚖️ Class Imbalance

- The official labeled dataset contains only a small number of studies compared with the full dataset.

- Additionally, the 12 abnormalities have substantially different positive frequencies.

- To account for this imbalance, the baseline model uses:

BCEWithLogitsLoss
        +
class-specific positive weights

This increases the contribution of positive examples from underrepresented abnormalities during training.

🔀 Study-Level Validation

To prevent slices or series from the same study appearing across both training and validation data, splitting is performed at the study level.

Current clean split:
```text
Officially labeled studies
        │
        ├── Training: 46 studies
        │
        └── Validation: 12 studies
```
- The validation studies are kept separate from the training data.

This provides a study-level evaluation rather than evaluating individual slices independently.

📝 Multilingual Radiology-Report Weak Supervision

The dataset contains radiology reports for many studies that do not have official abnormality labels.

The reports include multiple languages, requiring a conservative rule-based approach to extract usable supervision.

The weak-supervision pipeline:
```text
Radiology Report
       ↓
Language / terminology handling
       ↓
Abnormality-specific rules
       ↓
Confident Positive / Explicit Negative
       ↓
Weak Label
```
Three states are used for weak labels:

- 1     = Confident Positive
- 0     = Explicit Negative
- NaN   = Uncertain / Insufficient Evidence

Uncertain findings are not forced into positive or negative classes.

This is important because a radiology report may mention an abnormality without clearly establishing its presence or absence.

🧠 Weakly Supervised Labels

Weak labels were generated for the 4,349 studies without official abnormality labels.

The core weak-supervision targets currently include:

- ACL
- Medial Meniscus
- Lateral Meniscus
- PF OA
- Effusion
- Baker's Cyst

Additional abnormality extractors were also investigated, including:

- MCL
- Medial OA
- Lateral OA
- Synovitis
- Contusion
- Fracture

The weak-label pipeline generated confident labels while leaving uncertain cases masked rather than assigning potentially incorrect labels.

🎯 Masked Weak-Supervision Loss

Because not every radiology report provides enough information for every abnormality, weak labels contain missing/uncertain values.

Instead of treating these values as negative labels, a masked binary cross-entropy loss is used.

Conceptually:
```text
Weak Label
     │
     ├── Confident Positive → Used for training
     │
     ├── Explicit Negative   → Used for training
     │
     └── Uncertain           → Ignored
```
This prevents uncertain report interpretations from directly contributing incorrect gradients during training.

🔬 Weak-Supervised Model Experiment

- A separate weakly supervised 3D CNN experiment was performed using the automatically extracted report labels.

- The experiment demonstrated that the model could reduce training loss on the weak labels, but lower training loss did not necessarily translate into improved performance on the clean officially labeled validation set.

- This highlighted an important limitation of weak supervision:

- Learning from noisy or incomplete labels does not automatically produce better generalization to clinically verified labels.

- The weakly supervised model is therefore treated as an experimental component rather than a replacement for the clean-label baseline.

🔄 Combined Clean + Weak Training

A combined training strategy is being investigated in which:
```text
Officially Labeled Studies
        +
Weakly Labeled Studies
        ↓
Weighted Multi-label Training
        ↓
12-class 3D CNN
```
- The clean studies provide all 12 official labels, while the weak studies provide a subset of confident report-derived labels.

- The weak samples are assigned a lower training weight than the clean samples.

- This allows the experiment to investigate whether additional weakly labeled MRI studies can improve representation learning without allowing noisy supervision to dominate the training process.

🔍 Model Explainability

3D Grad-CAM was implemented to inspect where the CNN produces strong activation for a selected abnormality.
```text
The workflow is:

MRI Volume
    ↓
3D CNN
    ↓
Selected Abnormality
    ↓
Backward Pass
    ↓
Gradient-weighted Activations
    ↓
3D Grad-CAM
    ↓
Attention Visualization
```
Grad-CAM is used as an interpretability tool to investigate model attention and potential failure cases.

For example, attention can be visualized for predictions involving:

- Meniscal abnormalities
- ACL
- OA
- Effusion
- Other model outputs
  
Important limitation

Because axial, coronal, and sagittal volumes are represented as input channels rather than being explicitly co-registered into a common spatial coordinate system, the resulting Grad-CAM should be interpreted as a model attention visualization, not as precise anatomical lesion localization.

📈 Evaluation

The project evaluates performance using metrics suitable for multi-label classification.

Primary metrics include:

- ROC-AUC
- Precision
- Recall
- F1-score

Class-wise metrics are examined because the abnormalities have different frequencies and therefore different levels of classification difficulty.

Evaluation is performed at the study level.

For ROC-AUC, classes containing only a single label value in the validation split are treated as undefined rather than assigning an artificial score.

🧪 Experiments

The project currently includes several experimental stages:

- Experiment 1 — Baseline 3D CNN

A lightweight 3D CNN trained using officially labeled studies.
```text
Official Labels
      ↓
12-class 3D CNN
      ↓
Study-level Evaluation
```
- Experiment 2 — Multi-plane 3D CNN

A multi-plane architecture with separate processing branches for axial, coronal, and sagittal MRI data was investigated.

- Experiment 3 — Weak Supervision

Radiology reports were converted into conservative weak labels and used to train a six-target weakly supervised model.

- Experiment 4 — Clean + Weak Supervision

A combined 12-output model is being investigated using:
```text
46 Clean Training Studies
        +
3,290 Weakly Labeled Studies
        ↓
Weighted Training
        ↓
12-class 3D CNN
```
The clean validation set remains separate from the training data.

## 🛠️ Technologies & Libraries

### Programming Language
- Python

### Medical Imaging
- DICOM
- pydicom
- 3D MRI Processing
- Medical Image Preprocessing

### Machine Learning & Deep Learning
- PyTorch
- torchvision
- Scikit-learn
- NumPy
- Pandas

### Computer Vision & Visualization
- OpenCV
- Matplotlib
- Grad-CAM

### Development & Tools
- Kaggle
- Jupyter Notebook
- Git
- GitHub
📂 Project Structure
```text
RSNA-Knee-MRI-Abnormality-Detection/
│
├── rsna-knee.ipynb
├── README.md
└── ...
```
The main experimentation notebook contains the complete workflow from dataset exploration and DICOM preprocessing through model development and evaluation.

🚀 Workflow Summary

The overall pipeline can be summarized as:

                RSNA Knee MRI Dataset
                         │
                         ▼
                  Dataset Exploration
                         │
                         ▼
                   DICOM MRI Series
                         │
                         ▼
                 Metadata Inspection
                         │
                         ▼
                  Slice Ordering
                         │
                         ▼
                 3D Volume Construction
                         │
                         ▼
              MRI Series / Plane Selection
                         │
                         ▼
              Intensity Normalization
                         │
                         ▼
                   Volume Resizing
                         │
                         ▼
                 PyTorch Dataset
                         │
                         ▼
                    3D CNN
                         │
                         ▼
              12 Abnormality Outputs
                         │
                         ▼
              Study-Level Evaluation
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Error Analysis           Grad-CAM
              │                     │
              └──────────┬──────────┘
                         ▼
                 Model Improvement

The extended training pipeline additionally incorporates:

Radiology Reports
       ↓
Multilingual Report Processing
       ↓
Weak Label Extraction
       ↓
Masked Weak Supervision
       ↓
Clean + Weak Training Experiment
📌 Current Project Status

Status: In Progress

Completed:
- Dataset exploration
- Study and series analysis
- DICOM metadata inspection
- DICOM slice ordering
- 3D MRI volume reconstruction
- MRI intensity normalization
- Multi-plane MRI preprocessing
- Automated MRI series selection
- PyTorch dataset development
- Study-level train/validation split
- Lightweight 3D CNN baseline
- Class-imbalance-aware training
- Multi-plane CNN experiment
- Multilingual radiology-report weak supervision
- Conservative weak-label extraction
- Masked weak-supervision training
- 3D Grad-CAM attention visualization
  
Currently Investigating:
- Combined clean + weak supervised training
- Comparison of clean-label and weak-label training strategies
- Error analysis
- Model evaluation and improvement
  
Future Work:
- Further model architecture experimentation
- Improved weak-label extraction
- More robust validation strategies
- Threshold optimization
- Detailed per-abnormality error analysis
- Additional explainability analysis
- Final model selection and inference
  
📊 Key Dataset Statistics
Component	Count
MRI Studies	4,407
Imaging Series	24,371
Officially Labeled Studies	58
Studies Without Official Labels	4,349
Abnormality Categories	12
Clean Training Studies	46
Clean Validation Studies	12
Weakly Labeled Studies	3,290
Successfully Cached Weak Studies	3,288

⚠️ Limitations

Several limitations are considered during development:

- Only a small subset of studies contains official abnormality labels.
- Radiology-report-derived weak labels may contain noise or ambiguity.
- Different abnormalities have substantially different class frequencies.
- Some validation classes may contain only one class, making ROC-AUC undefined.
- MRI intensity values are not inherently standardized across studies.
- The current 3D CNN is intentionally lightweight due to CPU-only experimentation.
- Multi-plane inputs are not explicitly spatially co-registered.
- Grad-CAM therefore provides model attention information rather than definitive anatomical localization.
- Weak supervision is treated as an experimental augmentation strategy rather than ground-truth labeling.
  
📚 Dataset & Resources

The project is based on the RSNA Knee Abnormality Detection dataset.

The main experimentation notebook is available on Kaggle:

- RSNA Knee MRI Abnormality Detection: https://www.kaggle.com/code/anwesha7002/rsna-knee

👩‍💻 Authors
- Anwesha Chakraborty
- Dr Soumyajit Mukhopadhyay
