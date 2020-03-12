# Novelty-Prepared Few-Shot Classification
PyTorch implementations of the paper
> Chao Wang , Ruo-Ze Liu , Han-Jia Ye , Yang Yu. **Novelty-Prepared Few-Shot Classification**. ([https://arxiv.org/abs/2003.00497](https://arxiv.org/abs/2003.00497))


## Requirement
- python 3.6
- torch ==1.2.0
- torchvision == 0.4.0
- tqdm == 4.36.1



## Run

**Prepare data:**
CUB or MiniImagenet or FC100 (Fewshot-CIFAR100) or MiniImagenet->CUB (cross)

example: MiniImagenet

1. Download MiniImagenet dataset.

2. Extract subfolders to './filelists/miniImagenet/images'.

3. Run script "./filelists/miniImagenet/DataPreprocessing.sh".

4. Modify `data_dir` in the configs.py file to your corresponding path




**Train:**

python ./train.py --dataset miniImagenet --model HRNet --method SSL --train_aug



**Test:**

python ./save_features.py --dataset miniImagenet --model HRNet --method SSL --train_aug

python ./test.py --dataset miniImagenet --model HRNet --method SSL --train_aug


## Results
**CUB, mini-ImageNet and FC100 5-way Acc.**
| Model        |  CUB 5-way 1-shot   |  CUB 5-way 5-shot   | mini-ImageNet 5-way 1-shot | mini-ImageNet 5-way 5-shot| FC100 5-way 1-shot | FC100 5-way 5-shot|
| --------     | :-----: | :----: | :-----: | :----: |:-----: | :----: |
| SSL(ResNet-18)   | 74.05 ± 0.83%  | 89.92 ± 0.41%  | 60.98 ± 0.81% | 80.61 ± 0.49% | 47.43 ± 0.80% | 65.85 ± 0.75% |
| SSL(HRNet)       | 76.07 ± 0.82%  | 91.16 ± 0.37%  | 64.71 ± 0.83% | 83.23 ± 0.54% | 50.38 ± 0.80% | 69.32 ± 0.76% |




## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework:
CloserLookFewShot: https://github.com/wyharveychen/CloserLookFewShot
* Backbone:
HRNet: https://github.com/HRNet/HRNet-Image-Classification
* Dataset(FC100):
MTL: https://github.com/yaoyao-liu/meta-transfer-learning
