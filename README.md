# LASSO-MLP for stellar atmospheric parameters estimation

This repo contains the code and trained models for our paper *Estimation of stellar atmospheric parameters from LAMOST DR8 low-resolution spectra with 20≤SNR<30*.

### Requirements

- pandas

- numpy

- scikit-learn(>=0.15)



### Experimental data

-training data and test data：
```
Preprocessed_spectra.npy
```

-training label and test label：
```
LAMOST_APOGEE.csv
```


-estimation catalog
```
LASSO-MLP.csv
```



### Usage

- Training a new model:

  ```shell
  Jupyter Notebook LASSO_MLP.ipynb
  ```

### Citation


