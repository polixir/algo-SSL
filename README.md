# NPF
PyTorch implementations of the paper
> Chao Wang , Ruo-Ze Liu , Han-Jia Ye , Yang Yu. **Novelty-Prepared Few-Shot Classification**. ([arxiv](https://arxiv.org/abs/2003.00497))


## Requirement
- python 3.6
- torch ==1.2.0
- torchvision == 0.4.0
- tqdm == 4.36.1



## Run

**Prepare data:**
CUB or MiniImagenet or MiniImagenet->CUB (cross)

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
**CUB and mini-ImageNet 5-way Acc.**
| Model        |  CUB 5-way 1-shot   |  CUB 5-way 5-shot   | mini-ImageNet 5-way 1-shot | mini-ImageNet 5-way 5-shot|
| --------     | :-----: | :----: | :-----: | :----: |
| SSL(ResNet-18)   | 74.05 ± 0.83%  | 89.92 ± 0.41%  | 60.98 ± 0.81% | 80.61 ± 0.49% |
| SSL(HRNet)       | 76.07 ± 0.82%  | 91.16 ± 0.37%  | 64.71 ± 0.83% | 83.23 ± 0.54% |




## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework:
CloserLookFewShot:https://github.com/wyharveychen/CloserLookFewShot
* Backbone:
HRNet:https://github.com/HRNet/HRNet-Image-Classification