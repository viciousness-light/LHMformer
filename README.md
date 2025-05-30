# LHMformer:Long-range Historical Memory-EnhancedTransformer for Traffic Forecasting

#### Required Packages

```
pytorch==2.6.0
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



#### Training Commands

```bash

python experiments/train.py --cfg stid/${dataset}.py --gpus '0'


```

'${dataset}':
- METRLA
- PEMSBAY
- PEMS03
- PEMS04
- PEMS07
- PEMS08


