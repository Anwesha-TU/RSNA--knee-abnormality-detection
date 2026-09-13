# RSNA Knee MRI Abnormality Detection

A medical imaging project based on the **RSNA Knee Abnormality Detection** dataset, focused on analyzing knee MRI studies and developing a deep learning pipeline for multi-label abnormality classification.

## 📌 Project Overview

Knee MRI examinations contain multiple imaging series and DICOM slices that capture different anatomical planes and MRI sequences. The goal of this project is to build a study-level machine learning pipeline capable of identifying multiple knee abnormalities from MRI data.

The project currently focuses on understanding the dataset structure, exploring DICOM imaging data, visualizing MRI slices, and developing the preprocessing and deep learning pipeline.

## 🎯 Objective

Given a knee MRI study, predict the presence or absence of multiple abnormalities, including:

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

This is formulated as a **multi-label classification problem**, since a single MRI study may contain multiple abnormalities simultaneously.

## 📊 Dataset

The project uses the RSNA Knee Abnormality Detection dataset.

Current dataset exploration includes:

- **4,407 knee MRI studies**
- **24,371 imaging series**
- **58 studies with official abnormality labels**
- **4,349 studies with missing labels**
- **12 abnormality categories**

The dataset contains DICOM-based MRI imaging organized hierarchically:

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
The available imaging series include:

- Sagittal
- Coronal
- Axial

and different MRI sequence characteristics such as fluid-sensitive , proton density, T1- weighted and fat-suppressed imaging.

## 🔬 Current Progress

The project is currently in the medical image preprocessing and pipeline development stage.

### Completed
- Explored the RSNA Knee Abnormality Detection dataset.
- Analyzed 4,407 knee MRI studies and 24,371 imaging series.
- Identified 12 abnormality labels and examined the available labeled studies.
- Explored MRI series based on anatomical plane, fluid sensitivity, and fat suppression.
- Loaded and processed DICOM images using `pydicom`.
- Inspected important DICOM metadata including:
  - Magnetic field strength
  - Slice thickness
  - Pixel spacing
  - Image orientation
  - Image position
  - Instance number
- Implemented spatial/metadata-based DICOM slice ordering.
- Converted individual DICOM slices into 3D MRI volumes.
- Implemented percentile-based intensity normalization.
- Visualized original and normalized MRI slices.
- Started automated selection of suitable MRI series using dataset-provided imaging characteristics.

### In Progress
- Robust automated MRI series selection across studies.
- PyTorch dataset and preprocessing pipeline.
- Study-level multi-label CNN classification.
- Model training and evaluation.

### Planned
- Train a baseline CNN for multi-label abnormality classification.
- Evaluate model performance using appropriate multi-label classification metrics.
- Perform error analysis and investigate model improvements.
  
🛠️ Technologies & Libraries
- Python
- Medical Imaging
- DICOM
- pydicom
- NumPy
- Pandas
- PyTorch
- torchvision
- Scikit-learn
- Matplotlib
- Developed using - Kaggle, Jupyter Notebook

🧠 Planned Model Pipeline

The planned workflow is:
Knee MRI Study
       ↓
DICOM Series
       ↓
DICOM Slice Ordering
       ↓
Series Selection
       ↓
Image Preprocessing
       ↓
Normalization & Resizing
       ↓
PyTorch Dataset / DataLoader
       ↓
CNN Model
       ↓
Multi-Label Prediction
       ↓
12 Abnormality Probabilities

📈 Evaluation

- The model will be evaluated using metrics appropriate for multi-label classification, including: ROC-AUC, Precision, Recall, F1-score

- Class-wise performance will also be examined because the abnormalities have different label frequencies.

🚧 Project Status

Status: In Progress

The main experimentation notebook is available on Kaggle: 
RSNA Knee MRI Abnormality Detection
https://www.kaggle.com/code/anwesha7002/rsna-knee

👩‍💻 Author

- Anwesha Chakraborty
- Dr Soumyajit Mukhopadhyay

