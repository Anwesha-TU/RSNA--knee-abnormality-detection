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

and different MRI sequence characteristics such as fluid-sensitive and fat-suppressed imaging.

🔬 Current Work

The project is currently in the data exploration and preprocessing stage.

1. Dataset Exploration
- Examined the structure of train.csv and train_series.csv
- Analyzed the number of studies and imaging series
- Investigated anatomical planes and MRI sequence characteristics
- Examined the distribution of the available abnormality labels
- Identified the difference between labeled and unlabeled studies
  
2. DICOM Processing

- Using pydicom, MRI DICOM files are loaded and inspected to understand:

    - Image dimensions
    - Pixel data
    - DICOM metadata
    - MRI series information
    - Individual slice visualization
  
3. MRI Visualization

- Multiple DICOM slices from individual MRI series are visualized to inspect the anatomical content and understand the imaging structure before developing the model pipeline.

4. Preprocessing — In Progress

The next stage involves developing preprocessing steps for:

- Robust DICOM slice ordering
- MRI intensity normalization
- Image resizing/cropping
- Series selection
- Conversion of DICOM data into model-ready tensors
  
5. Deep Learning Pipeline — In Progress

A PyTorch-based pipeline is being developed for:

- Study-level dataset creation
- Train/validation splitting
- MRI image preprocessing
- Multi-label classification
- CNN-based abnormality prediction
  
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
- Visualization
- Matplotlib
Developed using - Kaggle, Jupyter Notebook

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

Anwesha Chakraborty
