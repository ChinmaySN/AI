# 🧠 CancerNet - Breast Cancer Classification with CNN
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChinmaySN/Breast-Cancer-Classification-with-CNN/blob/main/cancer.ipynb)


This project builds a Convolutional Neural Network (CNN) to classify breast cancer histology images as **Benign (0)** or **Malignant (1)** using the IDC_regular_ps50_idx5 dataset.

---

## 📂 Dataset

- Source: [Kaggle - IDC_regular_ps50_idx5](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- Description: 277,524 image patches (50×50) from 162 breast cancer slides
- Classes:
  - `0`: Benign (no IDC)
  - `1`: Malignant (IDC present)

> ⚠️ Dataset not included due to size — download it manually from Kaggle and extract it to a folder named `IDC_regular_ps50_idx5/`.

---

## 📊 Model Summary

| Metric     | Value     |
|------------|-----------|
| Model      | CancerNet (CNN) |
| Accuracy   | ~84.75%   |
| F1 Score   | 0.85      |
| Framework  | TensorFlow + Keras |
| Language   | Python 3  |
| Hardware   | Google Colab GPU |

---

## 🔬 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Google Colab
- NumPy, Matplotlib, Scikit-learn

---

## 🚀 How to Run

1. Clone this repo or open in Colab
2. Download the dataset from Kaggle
3. Place dataset in a folder: `dataset/IDC_regular_ps50_idx5/`
4. Run the notebook `cancer.ipynb`

---

## 🖼️ Predict New Images

You can upload a single image and run:

```python
predict_image('your_image.jpg', model)
