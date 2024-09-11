# Classification of Skin Lesions

## Description

The goal of this work was to perform supervised classification of skin lesions to 8 diagnostic classes. This project is part of [Télécom Paris Image Processing](https://www.telecom-paris.fr/fr/ingenieur/formation/2e-annee-orientation/image) course and its goal was to apply classical machine learning methods and convolutional neural networks (CNN) to classify medical images in a highly imbalanced dataset. In order to apply the classical machine learning methods, a segmentation pipeline to output the lesion area mask was implemented based on UNet CNN architecture.

Classical machine learning methods used

- K-Nearest Neighbours
- Suppport Vectors Classifier with BRF kernel
- Previous methods with oversampling (ADASYN)

Convolutional neural networks used for classification and segmentation:

- LeNet-5
- Inception-ResNet-v2
- UNet

## Dataset

The skin lesion images along with metadata were taken from ISIC dataset, available on https://challenge.isic-archive.com/data/.

## Requirements

### For classical machine learning methods

- Python 3.10
- NumPy 1.26.0
- Pandas 2.2.1
- Scikit-Image 0.21.0
- Scikit-Learn 1.4.2
- OpenCV 4.8.1
- Joblib 0.14.0
- Imbalanced-learn 0.12.3

### Additionals for deep learning methods

- PyTorch 2.3.0
- PyTorch-cuda 11.8
- Pillow 10.4.0
- TorchVision 0.18.0
- PyTorch Image Models (timm) 0.1.8 : https://huggingface.co/timm
- Tqdm 2.2.3

## Contributors

Paulo Roberto de Moura Júnior (me)

[High level description]()
