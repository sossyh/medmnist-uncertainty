# Knowing When the Model Doesn't Know

**Uncertainty Estimation for Medical Image Classification**

Final project for CSCI 6366: Neural Networks and Deep Learning, The George Washington University.

## Project Summary

Deep neural networks often give highly confident predictions even when they are wrong, or when an input looks nothing like their training data. In medical imaging, a model that is confidently wrong is dangerous, while a model that can say "I am not sure" and refer the case to a clinician is much safer.

This project asks: **which deep learning techniques let a network recognize when it should not be trusted?**

We train a small convolutional neural network (CNN) on chest X-rays (normal vs. pneumonia) and compare four ways of estimating how confident the model really is:

1. **Softmax baseline:** the model's raw maximum probability.
2. **Monte Carlo (MC) Dropout:** dropout stays on at test time, each image is run many times, and the spread of the predictions measures uncertainty.
3. **Deep Ensemble:** five independently trained networks whose predictions are averaged.
4. **Temperature scaling:** one learned temperature, fit on the validation set, that recalibrates the softmax.

Each method is tested on three kinds of data:

- the clean chest X-ray test set,
- the same test images with noise, blur, and reduced contrast at increasing severity,
- out-of-distribution images (breast ultrasound and retinal scans) that the model never saw in training.

We will report accuracy, Expected Calibration Error (ECE), negative log-likelihood (NLL), Brier score, AUROC for detecting out-of-distribution images, and accuracy versus coverage when the model abstains on its most uncertain cases. The final demo takes a single image and returns a prediction, an uncertainty score, and a "refer for review" flag.

## Team

| Name | GitHub Username |
|---|---|
| Sosna Worku Achamyeleh | [@sossyh](https://github.com/sossyh) |
| Sinchana Manjula Prabhakara | [@SINCHANA_USERNAME](https://github.com/SINCHANA_USERNAME) |

## Datasets

All datasets come from **MedMNIST v2**, a collection of standardized 28x28 medical images. They are free, public, and download automatically through the `medmnist` Python package.

| Dataset | Image type | Classes | Role in this project |
|---|---|---|---|
| PneumoniaMNIST | Pediatric chest X-ray | normal, pneumonia | Training, validation, and testing |
| BreastMNIST | Breast ultrasound | (not used) | Out-of-distribution test set |
| OCTMNIST | Retinal OCT scan | (not used) | Out-of-distribution test set |

**MedMNIST (processed versions used in this project)**

- Official dataset download (Zenodo): https://doi.org/10.5281/zenodo.10519652
- Project website: https://medmnist.com/
- Code and loader: https://github.com/MedMNIST/MedMNIST
- License: CC BY 4.0

**Original sources of the images**

- PneumoniaMNIST and OCTMNIST: Kermany, D., Zhang, K., and Goldbaum, M. (2018). *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images.* Mendeley Data. https://data.mendeley.com/datasets/rscbjbr9sj/3
- BreastMNIST: Al-Dhabyani, W., Gomaa, M., Khaled, H., and Fahmy, A. (2020). *Dataset of breast ultrasound images.* Data in Brief, 28, 104863. https://doi.org/10.1016/j.dib.2019.104863

## Repository Structure

```
.
├── README.md
├── requirements.txt
└── notebooks/
    └── 01_data_exploration_and_baseline.ipynb
```

## Current Progress

**Notebook 01: Data exploration and baseline** (`notebooks/01_data_exploration_and_baseline.ipynb`)

- Loads PneumoniaMNIST (train / validation / test) and the BreastMNIST and OCTMNIST test sets.
- Shows class balance per split and sample images from each dataset.
- Defines the model: a CNN with three convolution blocks (convolution, batch normalization, ReLU, max pooling, dropout) and a classifier with dropout. Dropout is built in because MC Dropout needs it.
- Trains the baseline with Adam and cross-entropy, keeping the weights with the best validation accuracy.
- Reports test accuracy, recall for each class, and a first look at softmax confidence on correct vs. wrong predictions and on never-seen images.

**Next steps:** MC Dropout, the deep ensemble, temperature scaling, the corrupted test sets, and the full evaluation (ECE, NLL, Brier, OOD AUROC, accuracy vs. coverage, reliability diagrams).

## How to Run

The easiest way is Google Colab:

1. Open the notebook in Colab (File > Open notebook > GitHub, then paste this repository's URL).
2. Set Runtime > Change runtime type > GPU.
3. Run all cells. The datasets download automatically on the first run.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration_and_baseline.ipynb
```

## References

1. Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML. https://arxiv.org/abs/1506.02142
2. Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NIPS. https://arxiv.org/abs/1612.01474
3. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML. https://arxiv.org/abs/1706.04599
4. Ovadia, Y. et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS. https://arxiv.org/abs/1906.02530
5. Hendrycks, D. and Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR. https://arxiv.org/abs/1610.02136
6. Yang, J. et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data, 10, 41. https://github.com/MedMNIST/MedMNIST
