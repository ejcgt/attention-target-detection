# CVPR 2020 - Detecting Attended Visual Targets in Video
![](misc/teaser.gif)

## Overview
This repo provides PyTorch implementation of our paper:
**'Detecting Attended Visual Targets in Video'**  [[paper]](https://arxiv.org/abs/2003.02501)

We present a state-of-the-art method for predicting attention targets from third-person point of view. The model takes head bounding box of a person of interest, and outputs an attention heatmap of that person.

We release our new dataset, training/evaluation code, a demo code, and pre-trained models for the two main experiments reported in our paper. Pleaser refer to the paper for details.


## Getting Started
The code has been verified on Python 3.5 and PyTorch 0.4. We provide a conda environment.yml file which you can use to re-create the environment we used. Instructions on how to create an environment from an environment.yml file can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Download our model weights using:
```
sh download_models.sh
```

## Quick Demo
You can try out our demo using the sample data included in this repo by running:
```
python demo.py
```

## Experiment on the GazeFollow dataset
### Dataset
We use the extended GazeFollow annotation prepared by [Chong et al. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/html/Eunji_Chong_Connecting_Gaze_Scene_ECCV_2018_paper.html), which makes an additional annotation to the original [GazeFollow dataset](http://gazefollow.csail.mit.edu/) regarding whether gaze targets are within or outside the frame. You can download the extended dataset from [here (image and label)](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0) or [here (label only)](https://www.dropbox.com/s/1mhgpu0x2w5yto6/gazefollow_extended_txtonly.zip?dl=0).

Please adjust the dataset path accordingly in config.py.
### Evaluation
Run:
```
python eval_on_gazefollow.py
```
to get the model's performance on the GazeFollow test set.
### Training
Run:
```
python train_on_gazefollow.py
```
to train the model. You can expect to see similar learning curves to [ours](https://tensorboard.dev/experiment/eDyILnKaSVa6efJXqTQkhg/).



## Experiment on the VideoAttentionTarget dataset
### Dataset
We created a new dataset, *VideoAttentionTarget*, with fully annotated attention targets in video for this experiment. Dataset details can be found in our paper. Download the VideoAttentionTarget dataset from [here](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0).  

Please adjust the dataset path accordingly in config.py.
### Evaluation
Run:
```
python eval_on_videoatttarget.py

```
to get the model's performance on the VideoAttentionTarget test set.
### Training
Run:
```
python train_on_videoatttarget.py
```
to do the temporal training.

## Citation
If you use our dataset and/or code, please cite
```bibtex
@inproceedings{Chong_2020_CVPR,
  title={Detecting Attended Visual Targets in Video},
  author={Chong, Eunji and Wang, Yongxin and Ruiz, Nataniel and Rehg, James M.},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

If you only use the extended GazeFollow annotations, please cite
```bibtex
@InProceedings{Chong_2018_ECCV,
author = {Chong, Eunji and Ruiz, Nataniel and Wang, Yongxin and Zhang, Yun and Rozga, Agata and Rehg, James M.},
title = {Connecting Gaze, Scene, and Attention: Generalized Attention Estimation via Joint Modeling of Gaze and Scene Saliency},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```


## References
We make use of the PyTorch ConvLSTM implementation provided by https://github.com/kamo-naoyuki/pytorch_convolutional_rnn.


## Contact
If you have any questions, please email Eunji Chong at eunjichong@gatech.edu.
