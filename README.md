# LHMformer:Long-range Historical Memory-EnhancedTransformer for Traffic Forecasting

#### LHMformer_arch

<img src="figures/LHMformer_arch.png" height="400"/>

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


#### Performance on Traffic Forecasting Benchmarks
<img src="figures/Performance.png" height="300"/>
<img src="figures/Performance (2).png" height="200"/> <img src="figures/Performance (3).png" height="200"/>

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


