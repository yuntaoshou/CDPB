# Unsupervised Graph Domain Adaptation with Dual Perturbation Branch for Survival Analysis

This is an official implementation of 'Unsupervised Graph Domain Adaptation with Dual Perturbation Branch for Survival Analysis' :fire:.

<div  align="center"> 
  <img src="https://github.com/yuntaoshou/MGLRA/blob/main/archi.png" width=100% />
</div>



## 🚀 Installation

```bash
certifi                  2024.8.30
charset-normalizer       3.4.0
contourpy                1.3.0
cycler                   0.12.1
ecos                     2.0.14
einops                   0.8.0
filelock                 3.13.1
fonttools                4.54.1
fsspec                   2024.2.0
huggingface-hub          0.26.2
idna                     3.10
Jinja2                   3.1.3
joblib                   1.4.2
kiwisolver               1.4.7
MarkupSafe               2.1.5
matplotlib               3.9.2
mpmath                   1.3.0
networkx                 3.2.1
numexpr                  2.10.1
numpy                    1.26.3
nvidia-cublas-cu12       12.4.5.8
nvidia-cuda-cupti-cu12   12.4.127
nvidia-cuda-nvrtc-cu12   12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.2.1.3
nvidia-curand-cu12       10.3.5.147
nvidia-cusolver-cu12     11.6.1.9
nvidia-cusparse-cu12     12.3.1.170
nvidia-ml-py             12.535.161
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.4.127
nvidia-nvtx-cu12         12.4.127
nvitop                   1.3.2
opencv-python            4.10.0.84
osqp                     0.6.7.post3
packaging                24.1
pandas                   2.2.3
pillow                   10.2.0
pip                      24.2
protobuf                 5.28.3
pyparsing                3.2.0
python-dateutil          2.9.0.post0
pytz                     2024.2
PyYAML                   6.0.2
qdldl                    0.1.7.post4
requests                 2.32.3
safetensors              0.4.5
scikit-learn             1.5.2
scikit-survival          0.23.0
scipy                    1.14.1
setuptools               75.1.0
six                      1.16.0
sympy                    1.13.1
tensorboardX             2.6.2.2
termcolor                2.5.0
threadpoolctl            3.5.0
timm                     1.0.11
torch                    2.5.1+cu121
torchaudio               2.5.1+cu121
torchvision              0.20.1+cu121
tqdm                     4.66.6
triton                   3.1.0
typing_extensions        4.9.0
tzdata                   2024.2
urllib3                  2.2.3
wheel                    0.44.0
```

## Training
```bash
nohup python train.py --source_dataset BLCA --source_dataset_dir "/data/ypq/BLCA_Features" --target_dataset_dir "/data/ypq/LGG_Features" --target_dataset LGG >> BLCA_LGG.log &
nohup python train.py --source_dataset BLCA --source_dataset_dir /data/ypq/BLCA_Features --target_dataset_dir /data/ypq/LUAD_Features --target_dataset LUAD >> BLCA_LUAD.log &
nohup python train.py --source_dataset BLCA --source_dataset_dir /data/ypq/BLCA_Features --target_dataset_dir /data/ypq/UCEC_Features --target_dataset UCEC >> BLCA_UCEC.log &



nohup python train.py --source_dataset LGG --source_dataset_dir /data/ypq/LGG_Features --target_dataset_dir /data/ypq/BLCA_Features --target_dataset BLCA >> LGG_BLCA.log &
nohup python train.py --source_dataset LGG --source_dataset_dir /data/ypq/LGG_Features --target_dataset_dir /data/ypq/LUAD_Features --target_dataset LUAD >> LGG_LUAD.log & 
nohup python train.py --source_dataset LGG --source_dataset_dir /data/ypq/LGG_Features --target_dataset_dir /data/ypq/UCEC_Features --target_dataset UCEC >> LGG_UCEC.log &



nohup python train.py --source_dataset LUAD --source_dataset_dir /data/ypq/LUAD_Features --target_dataset_dir /data/ypq/LGG_Features --target_dataset LGG >> LUAD_LGG.log &
nohup python train.py --source_dataset LUAD --source_dataset_dir /data/ypq/LUAD_Features --target_dataset_dir /data/ypq/BLCA_Features --target_dataset BLCA >> LUAD_BLCA.log &
nohup python train.py --source_dataset LUAD --source_dataset_dir /data/ypq/LUAD_Features --target_dataset_dir /data/ypq/UCEC_Features --target_dataset UCEC >> LUAD_UCEC.log &


nohup python train.py --source_dataset UCEC --source_dataset_dir /data/ypq/UCEC_Features --target_dataset_dir /data/ypq/LGG_Features --target_dataset LGG >> UCEC_LGG.log &
nohup python train.py --source_dataset UCEC --source_dataset_dir /data/ypq/UCEC_Features --target_dataset_dir /data/ypq/LUAD_Features --target_dataset LUAD >> UCEC_LUAD.log & 
nohup python train.py --source_dataset UCEC --source_dataset_dir /data/ypq/UCEC_Features --target_dataset_dir /data/ypq/BLCA_Features --target_dataset BLCA >> UCEC_BLCA.log &
```

