# Covid19--Image-Classification

This project uses deep learning (CNNs) to classify chest X-ray images into three categories:
- COVID-19
- Viral Pneumonia
- Normal

The goal is to assist radiologists in identifying COVID-19 from lung X-rays, providing faster and more accurate diagnoses.

---

## Dataset

- `CovidImages.npy` – Numpy array of X-ray images
- `CovidLabels.csv` – Corresponding labels for each image

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV
- Scikit-learn
- Seaborn, Matplotlib

---

## Project Highlights

- Built and trained CNN models from scratch
- Used data preprocessing and augmentation
- Compared baseline model with improved model using `ReduceLROnPlateau`
- Visualized performance using:
  - Accuracy curves
  - Confusion matrices
  - Classification reports

---

## Sample Results

| True Label | Predicted Label |
|------------|-----------------|
| COVID-19   | COVID-19        |
| Normal     | Normal          |
| Viral Pneumonia | Viral Pneumonia |

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/COVID19-Image-Classifier.git
   cd COVID19-Image-Classifier
