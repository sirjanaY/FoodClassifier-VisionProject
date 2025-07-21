![](UTA-DataScience-Logo.png)

# Food Image Classification with Transfer Learning

This project uses transfer learning on a subset of the [Food11 dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) to classify food images into 5 categories using various deep learning models.

---

##  Overview

We aim to classify food images using pre-trained models through transfer learning. Starting from a large dataset (Food11), we selected 5 food classes and reduced each to under 100 images for quick experimentation. We compared:
- A custom CNN baseline
- The same model with data augmentation
- A ResNet50 transfer learning model

Models are evaluated using accuracy and ROC curves.

---

##  Data

- **Source**: [Food11 Dataset on Kaggle](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
- **Classes Used**: `Dairy`, `Sugar`, `Protein`, `Mixed`
- **Format**: Images in class-specific folders
- **Split**: 80% Training, 20% Validation

---

## Preprocessing

- Selected 5 classes from the original dataset
- Trimmed each class to ≤100 images
- Normalized pixel values
- Applied one-hot encoding for labels
- Visualized images to confirm correct loading

---

##  Models

### 1. Baseline CNN
- Simple convolutional layers with BatchNorm and MaxPooling
- ~88% ROC AUC

### 2. Baseline + Augmentation
- Added: Random Flip, Rotation, Zoom
- ~91% ROC AUC

### 3. ResNet50 (Transfer Learning)
- Pretrained ResNet50 backbone
- ~94% ROC AUC

---

##  Training & Evaluation

- **Epochs**: 8
- **Batch Size**: 16
- **Optimizer**: Adam (LR: 3e-4)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, ROC-AUC

###  Visuals
- Training/validation curves in each notebook
- ROC curves in `CompareModels.ipynb`

---

##  Performance Summary

| Model              | Avg. ROC-AUC | Notes                          |
|-------------------|--------------|--------------------------------|
| Baseline CNN       | ~0.88        | Basic convolutional model      |
| Augmented Model    | ~0.91        | Better generalization          |
| ResNet50 Transfer  | ~0.94        | Best accuracy with fewer epochs|

---

## Reproduce This Project

1. Download and extract [Food11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
2. Run `DataLoader.ipynb` to clean & load subset
3. Train models using:
   - `TrainBaseModel.ipynb`
   - `TrainBaseModelAugmentation.ipynb`
   - `Train-ResNet.ipynb`
4. Evaluate:
   - `CompareAugmentation.ipynb`
   - `CompareModels.ipynb`

---

##  Files Overview

| File Name                  | Purpose                                         |
|---------------------------|-------------------------------------------------|
| `DataLoader.ipynb`        | Load and preprocess dataset                     |
| `TrainBaseModel.ipynb`    | Simple CNN model training                       |
| `TrainBaseModelAugmentation.ipynb` | Training with image augmentation       |
| `Train-ResNet.ipynb`      | Transfer learning using ResNet50               |
| `CompareAugmentation.ipynb` | ROC curve: baseline vs augmented             |
| `CompareModels.ipynb`     | Compare all models with ROC curves             |

---

##  Requirements

```bash
pip install tensorflow numpy matplotlib scikit-learn
