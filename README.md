# prediction_scripts
Code for INLA method and REDS, as used in bachelor thesis.

Both `data` and `scripts` are split into `satellite` (the MODIS data set and and prediction methods for 2D) and `simulated_1D` (a simulated Gaussian process and the same methods implemented for 1D):


```
├── data
│   ├── satellite
│   │   ├── test.csv
│   │   └── train.csv
│   └── simulated_1D
│       ├── 1D_test_1.csv
│       └── 1D_train_1.csv
├── LICENSE
├── README.md
└── scripts
    ├── satellite
    │   ├── inla.R
    │   └── REDS.py
    └── simulated_1D
        └── inla1D.R
```