# Knowing When the Model Doesn't Know
### Uncertainty Estimation for Medical Image Classification

Final project for **CSCI 6366: Neural Networks and Deep Learning**, The George Washington University.

## Project Summary

Deep neural networks can produce highly confident predictions even when their predictions are incorrect or when they encounter inputs that differ significantly from their training data. In medical imaging, such overconfidence can be particularly dangerous. A model that confidently produces an incorrect diagnosis may pose a risk, whereas a model capable of recognizing its uncertainty and referring the case to a clinician provides a safer approach.

This project investigates the following research question:

**Which deep learning techniques enable a neural network to recognize when its predictions should not be trusted?**

To explore this question, we train a convolutional neural network (CNN) on chest X-ray images to distinguish between normal cases and pneumonia. We then compare four approaches to predictive uncertainty estimation:

1. **Softmax Baseline:** Uses the model's maximum softmax probability as a measure of prediction confidence.
2. **Monte Carlo (MC) Dropout:** Keeps dropout active during inference, evaluates each image multiple times, and measures uncertainty based on the variation in predictions.
3. **Deep Ensemble:** Trains five independently initialized networks and combines their predictions through averaging.
4. **Temperature Scaling:** Learns a single temperature parameter using the validation set to recalibrate the model's softmax probabilities.

Each approach is evaluated across three data conditions:

- **Clean test data:** The original chest X-ray test set.
- **Corrupted test data:** The same test images subjected to noise, blur, and reduced contrast at increasing severity levels.
- **Out-of-distribution (OOD) data:** Breast ultrasound and retinal scans that the model has never encountered during training.

The evaluation will include accuracy, Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Brier score, AUROC for out-of-distribution detection, and accuracy versus coverage when the model abstains from making predictions on its most uncertain cases.

The final demonstration will accept a single medical image and return a predicted class, an uncertainty score, and a flag indicating whether the case should be referred for further clinical review.

## Team Members

| Name | GitHub Username |
|---|---|
| Sosna Worku Achamyeleh | [@sossyh](https://github.com/sossyh) |
| Sinchana Manjula Prabhakara | [@sinchanaprabhakar](https://github.com/sinchanaprabhakar) |

## Datasets

All datasets used in this project are obtained from **MedMNIST v2**, a collection of standardized 28 × 28 medical images. These datasets are publicly available and can be downloaded automatically using the `medmnist` Python package.

| Dataset | Image Type | Classes | Role in This Project |
|---|---|---|---|
| PneumoniaMNIST | Pediatric chest X-ray | Normal, Pneumonia | Training, validation, and testing |
| BreastMNIST | Breast ultrasound | Not used | Out-of-distribution test set |
| OCTMNIST | Retinal OCT scan | Not used | Out-of-distribution test set |

### MedMNIST (Processed Versions Used in This Project)

- **Official dataset download (Zenodo):** [https://doi.org/10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652)
- **Project website:** [https://medmnist.com/](https://medmnist.com/)
- **Source code and data loader:** [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST)
- **License:** CC BY 4.0

### Original Sources of the Images

- **PneumoniaMNIST and OCTMNIST:** Kermany, D., Zhang, K., and Goldbaum, M. (2018). *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images.* Mendeley Data. [https://data.mendeley.com/datasets/rscbjbr9sj/3](https://data.mendeley.com/datasets/rscbjbr9sj/3)

- **BreastMNIST:** Al-Dhabyani, W., Gomaa, M., Khaled, H., and Fahmy, A. (2020). *Dataset of breast ultrasound images.* Data in Brief, 28, 104863. [https://doi.org/10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)

## Repository Structure

```text
.
├── README.md
├── requirements.txt
└── notebooks/
    └── 01_data_exploration_and_baseline.ipynb
```

## Current Progress

### Notebook 01: Data Exploration and Baseline

**File:** `notebooks/01_data_exploration_and_baseline.ipynb`

The initial notebook covers the following stages of the project:

- Loads PneumoniaMNIST training, validation, and test datasets, along with the BreastMNIST and OCTMNIST test sets.
- Examines class distributions across the dataset splits and visualizes sample images from each dataset.
- Defines the baseline CNN architecture, consisting of three convolutional blocks with convolution, batch normalization, ReLU activation, max pooling, and dropout layers, followed by a classifier with dropout. Dropout is incorporated into the architecture to support the implementation of MC Dropout.
- Trains the baseline model using the Adam optimizer and cross-entropy loss, retaining the model weights that achieve the best validation accuracy.
- Evaluates test accuracy, class-wise recall, and an initial comparison of softmax confidence for correct and incorrect predictions, as well as predictions on previously unseen images.

### Next Steps

The remaining implementation and evaluation stages include:

- Monte Carlo (MC) Dropout.
- Deep Ensemble implementation.
- Temperature Scaling.
- Generation of corrupted test datasets.
- Comprehensive evaluation using ECE, NLL, Brier score, OOD AUROC, and accuracy versus coverage.
- Reliability diagram generation.

## How to Run

### Option 1: Google Colab (Recommended)

1. Open Google Colab.
2. Navigate to **File → Open notebook → GitHub** and paste the repository URL.
3. Select **Runtime → Change runtime type** and choose **GPU**.
4. Execute all notebook cells. The datasets will be downloaded automatically during the first execution.

### Option 2: Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/01_data_exploration_and_baseline.ipynb
```

## References

1. Gal, Y. and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML. [https://arxiv.org/abs/1506.02142](https://arxiv.org/abs/1506.02142)

2. Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NIPS. [https://arxiv.org/abs/1612.01474](https://arxiv.org/abs/1612.01474)

3. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)

4. Ovadia, Y. et al. (2019). *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS. [https://arxiv.org/abs/1906.02530](https://arxiv.org/abs/1906.02530)

5. Hendrycks, D. and Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR. [https://arxiv.org/abs/1610.02136](https://arxiv.org/abs/1610.02136)

6. Yang, J. et al. (2023). *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.* Scientific Data, 10, 41. [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST)
