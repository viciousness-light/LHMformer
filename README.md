# LHMformer:Long-range Historical Memory-EnhancedTransformer for Traffic Forecasting

#### Required Packages


```bash
# Install Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
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


