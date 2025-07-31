![UTA-DataScience-Logo](UTA-DataScience-Logo.png)

#  Food Image Classification with Transfer Learning

This project implements a complete image classification pipeline to categorize food images using deep learning techniques. I compare three different modeling approaches under constrained data conditions to understand how techniques like augmentation and transfer learning affect model performance when data is limited. The best-performing model achieved an **ROC-AUC of ~0.98**, demonstrating that efficient techniques can deliver strong results without large datasets.

---

##  Project Overview

###  Objective

To build and compare the performance of:
- A **simple CNN** baseline
- The same CNN with **data augmentation**
- A **ResNet50**-based **transfer learning model**

By restricting the dataset to **4 classes** and ~115 images/class, the focus was on **performance optimization under limited resources** — a common real-world scenario in early-stage projects or research prototyping.

---

##  Why This Project Matters

- Food image classification has real-world relevance in **fitness tracking, calorie estimation, and health tech**.
- It helps demonstrate **how far we can push performance using minimal data** and well-chosen architectures.
- The project allowed me to **practice modular deep learning pipeline construction**, experiment with **data augmentation**, and **apply transfer learning** to evaluate generalization.

---

##  Tech Stack

| Tool            | Why It's Used                                                   |
|-----------------|------------------------------------------------------------------|
| **Python**      | Versatile language for data science                             |
| **TensorFlow**  | Industry-standard for building and training deep learning models |
| **NumPy**       | Efficient numerical operations and array manipulation            |
| **Matplotlib**  | Visualization of training metrics, image samples, and ROC curves|
| **scikit-learn**| ROC-AUC scoring and data splitting tools                         |

---

##  Dataset Summary

- **Source**: [Kaggle Food11 Dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
- **Original Categories**: 11
- **Chosen Subset**:  
  - `Dairy` → Renamed to `Dairy`  
  - `Dessert` → Renamed to `Sugar`  
  - `Meat` → Renamed to `Protein`  
  - `Soup` → Renamed to `Mixed`

I chose these classes for **distinct visual characteristics** to help models learn clearly separable features. Limiting each class to ~115 images mimics real-world constraints (e.g., few labeled samples in medical imaging or food tracking apps).

---

##  Preprocessing Strategy

| Step                       | Reasoning                                                                 |
|----------------------------|---------------------------------------------------------------------------|
| Selected 4 classes         | Reduce complexity & speed up experimentation                              |
| ≤115 images/class          | Simulate a low-data environment                                           |
| Normalized pixel values    | Ensures faster convergence and prevents gradient explosion                |
| One-hot encoding labels    | Required for multi-class classification using `categorical_crossentropy` |
| Visual inspection          | Verifies data cleanliness and confirms accurate labeling                  |

![img4](img4.png)

---

##  Models Trained and Rationale

### 1. **Baseline CNN (MobileNetV2-inspired)**

- Built a simple Conv2D network with MaxPooling, Dropout, and BatchNorm layers
- Chosen for speed and simplicity
- Good for learning core image features from scratch

**Why?**  
To establish a baseline and understand how well a small CNN can perform without additional data handling. This also helps benchmark improvements from augmentation or transfer learning.

→ **ROC-AUC: ~0.98**

---

### 2. **CNN with Data Augmentation**

- Augmented input images using real-time transformations:
  - Horizontal flip
  - Random zoom
  - Random rotation

**Why?**  
To improve generalization and reduce overfitting. Augmentation forces the model to learn **invariant features** and prevents it from memorizing small datasets.

→ **ROC-AUC: ~0.98** (slightly better generalization)

![img2](img2.png)

---

### 3. **ResNet50 (Transfer Learning)**

- Used pretrained ResNet50 from ImageNet
- Only trained the final classification head

**Why?**  
To evaluate if pre-learned "universal image features" (like textures, edges, shapes) can help classify food images with limited labeled data. This approach often outperforms training from scratch on small datasets.

**Result:**  
→ ROC-AUC: ~0.70  
**Reason:** Underperformed due to likely overfitting and lack of tuning. Pretrained models often need **careful fine-tuning** or **more data** to adapt well.

---

## Training Configuration

| Hyperparameter | Value                     | Rationale                                                  |
|----------------|---------------------------|------------------------------------------------------------|
| Epochs         | 20                        | Balanced speed with model convergence                      |
| Batch Size     | 16                        | Good trade-off for memory use and gradient stability       |
| Optimizer      | Adam (lr = 3e-4)          | Adaptive learning for stable training                      |
| Loss           | Categorical Crossentropy  | Multi-class classification objective                       |
| Metric         | Accuracy, ROC-AUC         | Evaluates performance holistically                         |

---

##  Results Summary

| Model                 | ROC-AUC      | Insights                                |
|-----------------------|--------------|------------------------------------------|
| Baseline CNN          | ~0.98        | Strong baseline; learns features well    |
| Augmented CNN         | ~0.98        | More robust to overfitting               |
| ResNet50 Transfer     | ~0.70        | Underfit; highlights tuning importance   |

![img1](img1.png)

---

##  Key Observations

- **Data augmentation** slightly improved generalization without changing the architecture  
- **Transfer learning is not always better** — especially when the dataset is small and not similar to the pretraining domain  
- Even simple CNNs can perform **exceptionally well with the right preprocessing**  
- The pipeline structure makes it easy to test other architectures or configurations

---

##  Future Extensions

- **Full Dataset Use**: Scale up to all 11 classes and larger sample sizes  
- **Architecture Swaps**: Try EfficientNet, Vision Transformers (ViT), or MobileNetV3  
- **Preprocessing Enhancements**: Apply segmentation to isolate food from background  
- **Multilabel Support**: Handle real-world composite dishes with overlapping categories  
- **Web App Interface**: Add prediction UI with Streamlit or Flask  
- **AutoML**: Use tools like Keras Tuner or AutoKeras for hyperparameter optimization

---

## Reproducibility Steps

### 1. Download & Prepare Data
- From: [Kaggle: Food11 Dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
- Select 4 folders: `Dairy`, `Dessert`, `Meat`, `Soup`
- Rename and reorganize into: `Dairy`, `Sugar`, `Protein`, `Mixed`

### 2. Run Notebooks in This Order

| Notebook                          | Purpose                                     |
|----------------------------------|---------------------------------------------|
| `DataLoader.ipynb`               | Load, preprocess, normalize, label encode   |
| `TrainBaseModel.ipynb`           | Train CNN from scratch                      |
| `TrainBaseModelAugmentation.ipynb` | Add augmentation and retrain              |
| `Train-ResNet.ipynb`             | Transfer learning with ResNet50             |
| `CompareAugmentation.ipynb`      | ROC-AUC plots for baseline vs augmented     |
| `CompareModels.ipynb`            | Final evaluation + model comparison table   |
| `TestModel.ipynb`                | View predictions on test samples            |

 *Example output:*  
![img3](img3.png)

---

##  Setup & Installation

Install required libraries using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
