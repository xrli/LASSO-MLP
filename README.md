# LASSO-MLP for stellar atmospheric parameters estimation

This repo contains the code and trained models for our paper *Estimation of stellar atmospheric parameters from LAMOST DR8 low-resolution spectra with 20≤SNR<30*.

### Requirements

- Presto(<https://github.com/scottransom/presto>)

- numpy

- scipy

- pandas

- scikit-learn(>=0.15)

- tensorflow(>=1.8.0)

- keras

- keras_metrics

  **Note: The Python version depends on what version of python is Presto installed on. In other words, the code of CCNN can run both in Python 2 and 3 except for the preprocessing for data.**

### Experimental data

-training data and test data

-estimation catalog


### Usage

- Training a new model:

  ```shell
  python training.py
  ```


### Citation
