# LHMformer:Long-range Historical Memory-EnhancedTransformer for Traffic Forecasting

#### Required Packages

```
pytorch>=2.4.0
easy-torch
easydict
packaging
setproctitle
pandas
scikit-learn
tables
sympy
openpyxl
setuptools
numpy
```

#### Data Preparation

You can download the `dataset.zip` file from [Google Drive](https://drive.google.com/file/d/19c8YJDuRIQEsgPWSP_UcIVF_Vqq0fRFV/view?usp=drive_link)

#### Performance on Traffic Forecasting Benchmarks

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems04)](https://paperswithcode.com/sota/traffic-prediction-on-pems04?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems07)](https://paperswithcode.com/sota/traffic-prediction-on-pems07?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems08)](https://paperswithcode.com/sota/traffic-prediction-on-pems08?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-metr-la)](https://paperswithcode.com/sota/traffic-prediction-on-metr-la?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems-bay)](https://paperswithcode.com/sota/traffic-prediction-on-pems-bay?p=spatio-temporal-adaptive-embedding-makes)

![perf1](https://github.com/XDZhelheim/STAEformer/assets/57553691/8049bce2-9bc2-4248-a911-25468e9bbab4)

<img width="600" alt="image" src="https://github.com/XDZhelheim/STAEformer/assets/57553691/abf009aa-b145-451c-aff6-27031d60a612">

#### Training Commands

```bash

python experiments/train.py --cfg LHMformer/${dataset}.py --gpus '0'


```

'${dataset}':
- METRLA
- PEMSBAY
- PEMS03
- PEMS04
- PEMS07
- PEMS08


