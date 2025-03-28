# VisNet 
## VisNet: A Human Visual System Inspired Lightweight Dual-Path Network for Medical Images Denoising
Efficiently and accurately removing noise from medical images is crucial for clinical diagnosis. Nevertheless, most deep learning-based medical images denoising methods are highly complex and inaccurate in preserving the edge and shape of different organs, resulting in suboptimal denoising performance. In our study, we propose a Human Visual System Inspired Lightweight Dual-Path Network for medical images denoising (VisNet), which can efficiently and accurately remove noise from different types of medical images. Specifically, to simulate the mechanism in the visual system where magnocellular and parvocellular pathways capture significant and subtle noise, respectively, we design a dual-path multi-scale perception module. Then, to simulate the function of the primary visual cortex, we propose an edge detection and shape adaptation module to preserve the structural information of the medical images. Finally, inspired by dorsal and ventral pathways, a spatial-semantic information extraction module is designed to enhance the main semantic information in the image through the interactive fusion between the spatial and semantic pathways. Experimental results demonstrate that VisNet achieves superior performance across three medical datasets compared to nine existing baselines, while maintaining minimal computational complexity. (Params=0.15, FLOPs=16.41). In addition, for brain tumor classification, using denoised images of VisNet as input significantly improves accuracy (87.5% vs 96.7%) and achieves performance comparable to noise-free images.

![image](https://github.com/user-attachments/assets/64c7b0db-6075-4af2-8e41-ca5df0d40b9a)

## 1. CT denoising
### 1.1 convert dicom file to numpy array
```
python prep.py
```
### 1.2 Training
```
python main.py --mode='train'
```
### 1.3 Testing
```
python main.py --mode='test'
```

## 2. X-ray and MRI denoising
### 2.1 Training 
```
DPNtrainJSRT.py # training for X-ray
```

```
DPNtrainMRI15.py # training for MRI, noise level = 15
DPNtrainMRI50.py # training for MRI, noise level = 50
```
### 2.2 Testing 
```
DPNpredictJSRT.py # testing for X-ray
```
```
DPNpredictMRI15GP.py # testing for MRI, noise level = 15
DPNpredictMRI50GP.py # testing for MRI, noise level = 50
```


