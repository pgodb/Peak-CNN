# Peak-CNN

This repository contains code associated with the paper:  [Godbersen, P., Schanz, D. & Schröder, A. Peak-CNN: improved particle image localization using single-stage CNNs. Exp Fluids 65, 153 (2024)](https://doi.org/10.1007/s00348-024-03884-z)

We provide a python implementation of Peak-CNN using Tensorflow2/Keras in `PeakCNN_funcs.py` as well as a small testcase in `testcase.ipynb` showing training and application of the model to synthetic data.

If you find this work useful, please cite the coresponding paper referenced in Citation.bib


Peak-CNN itself depends only on Tensorflow and Numpy, the testcase notebook additionaly uses h5py and matplotlib.
Their respective licenses can be found under:

- Tensorflow (LGPL):  https://github.com/tensorflow/tensorflow/blob/master/LICENSE
- Numpy (custom):   https://github.com/numpy/numpy/blob/main/LICENSE.txt
- h5py (BSD 3-Clause): https://github.com/h5py/h5py/blob/master/LICENSE
- Matplotlib (custom):   https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE