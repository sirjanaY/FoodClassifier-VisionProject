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

Food image classification is a valuable real-world application with relevance in areas like fitness tracking, calorie estimation, and personalized health tech. This project explores how deep learning models can be effectively applied even in low-data environments, where collecting large labeled datasets may not be feasible. By carefully selecting architectures and applying strategies like data augmentation and transfer learning, I was able to achieve strong performance while keeping the pipeline lightweight and efficient. Beyond the technical results, this project also served as a hands-on opportunity to build a modular, reusable deep learning workflow and deepen my understanding of how different modeling choices affect generalization.

---
![img4](img4.png)

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


*To streamline the task and reduce training time, I selected four visually distinct food categories from the original Food11 dataset. I limited each class to a maximum of 115 images to simulate a realistic low-data scenario, which is common in many practical applications.

*The images were rescaled from 0–255 to 0–1 to normalize pixel values, which helps stabilize training and ensures faster convergence. I also one-hot encoded the labels to prepare them for multi-class classification using the categorical_crossentropy loss function.

*Finally, I manually inspected the images to confirm correct labeling and remove any visually misleading or corrupted samples. These steps helped ensure that the models were trained on clean, balanced, and well-prepared data.

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

### 2. ** Data Augmentation**

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

**Set the number of epochs to 20**: 
Training for 20 epochs provided a good balance between giving the model enough time to learn and avoiding overfitting. Since the dataset was small, longer training could have led to memorization rather than generalization.

**Use a batch size of 16** :
A moderate batch size like 16 ensured stable gradient updates while being small enough to fit within CPU memory. It also helped the model generalize better than using very large or very small batch sizes.

**Choose the Adam optimizer with a learning rate of 3e-4**: 
Adam was selected for its adaptive learning rate capabilities, which adjust the step size for each parameter during training. This helps achieve faster and more stable convergence. The learning rate of 3e-4 was chosen to make learning efficient without overshooting the optimal weights.

**Apply categorical crossentropy as the loss function**: 
Since this is a multi-class classification problem with one-hot encoded labels, categorical crossentropy was the appropriate loss function. It helps the model learn probabilities for each class and penalizes incorrect predictions more strongly.

**Monitor accuracy and ROC-AUC as evaluation metrics** : 
Accuracy gave a straightforward measure of overall correctness, while ROC-AUC offered a deeper view into how well the model distinguished between different classes. This was especially important to assess model confidence and class separation in a low-data setting.

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
![img3](img3.png)

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


 `DataLoader.ipynb`                Load, preprocess, normalize, label encode   
 `TrainBaseModel.ipynb`            Train CNN from scratch                      
 `TrainBaseModelAugmentation.ipynb`  Add augmentation and retrain              
 `Train-ResNet.ipynb`             Transfer learning with ResNet50             
 `CompareAugmentation.ipynb`      ROC-AUC plots for baseline vs augmented    
`CompareModels.ipynb`             Final evaluation + model comparison table   
`TestModel2.ipynb`                 View predictions on test samples            



---

##  Setup & Installation

Install required libraries using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
