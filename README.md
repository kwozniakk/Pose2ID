<h1 align='center'>From Poses to Identity: Training-Free Person Re-Identification via Feature Centralization</h1>

<div align='center'>
    <a href='https://github.com/yuanc3' target='_blank'>Chao Yuan</a><sup>*</sup>&emsp;
    <a href='https://github.com/zhangguiwei610' target='_blank'>Guiwei Zhang</a><sup>*</sup>&emsp;
    <a href='https://github.com/maxiaoxsi' target='_blank'>Changxiao Ma</a><sup></sup>&emsp;
    <a  target='_blank'>Tianyi Zhang</a><sup></sup>&emsp;
    <a  target='_blank'>Guanglin Niu</a><sup></sup>
</div>

<div align='center'>
Beihang University
</div>
<br>
<div align='center'>
    <a href='https://'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://arxiv.org/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>




![Image](docs/visualization.png "Teaser")

Our **Identity-Guided Pedestrian Generation** model obtain high-quality images with diverse poses, ensuring identity consistency even in complex scenarios such as **infrared**, and **occlusion**.


## &#x1F4E3; Updates
* [2025.02.27] 🔥 **Pose2ID** is accepted by CVPR 2025.


## ⚒️ Installation

There are two parts of our project: **Identity-Guided Pedestrian Generation (IPG)** and **Neighbor Feature Centralization (NFC)**.

**IPG** using generated pedestrian images to centralize features. Using simple codes could implement:

```bash
    '''
    normal reid feature extraction to get 'feats'
    '''
    feats_ipg = torch.zeros_like(feats)
    # fuse features of generated positive samples with different poses
    for i in range(num_poses):
        feats_ipg += reid_model(feats_pose[i]) # Any reid model
    eta = 1 # control the impact of generated images (considering the quality)

    # centralize features and normalize to original distribution
    feats = torch.nn.functional.normalize(feats + eta * feats_ipg, dim=1, p=2) # L2 normalization
    '''
    compute distance matrix or post-processing like re-ranking
    '''
```

**NFC** explores each sample's potential positive samples from its neighborhood. It can also implement with few lines:

```bash
  from NFC import NFC
  feats = NFC(feats, k1 = 2, k2 = 2)
```



### Download the Codes

```bash
  git clone https://github.com/yuanc3/Pose2ID
  cd Pose2ID/IPG
```

### Python Environment Setup
Create conda environment (Recommended):

```bash
  conda create -n IPG python=3.9
  conda activate IPG
```

Install packages with `pip`
```bash
  pip install -r requirements.txt
```

### Download pretrained weights
Download our model weights from [Google Drive](https://drive.google.com/drive/folders/1q5MNFMB1FV74Xy2vPo43k3tbOthQijDS?usp=sharing) and put them in the `pretrained` directory.

The **pretrained** are organized as follows.

```
./pretrained/
├── denoising_unet.pth
├── reference_unet.pth
├── IFR.pth
├── pose_guider.pth
└── transformer_20.pth
```

Some of the pretrained weights are from the following repositories:
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

### Inference
Run the python inference script. It will generate with poses in the `standard_poses` for each reference image in `ref`. The output images will be saved in the `output`.


```bash
  python inference.py --ckpt_dir pretrained --pose_dir standard_poses --ref_dir ref --out_dir output
```

where 
`--ckpt_dir` is the directory of pretrained weights,\
`--pose_dir` is the directory of target poses, \
`--ref_dir` is the directory of reference images, \
`--out_dir` is the directory of output images.


## 📝 Release Plans

|  Status  | Milestone                                                                | ETA |
|:--------:|:-------------------------------------------------------------------------|:--:|
|    🚀    | Training codes       | TBD |


## 📒 Citation

If you find our work useful for your research, please consider citing the paper :

```

```