# MC-CNN-Chainer
This repository provides a Chainer implementation and pre-trained models of MC-CNN proposed in
```
Jure Zbontar and Yann LeCun:
"Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches",
Journal of Machine Learning Research, vol. 17, pages 1-32, 2016.
```
Our pre-trained models are produced by importing network weights from their original pre-trained models in https://github.com/jzbontar/mc-cnn. Our imlementation uses much less GPU memories for computing matching costs.

## Remarks
+ We only provide code for inference. Training code is not included.
+ We only provide code of convolutional neural networks. Code for subsequent cross-based and SGM-based cost aggregation parts is not included.
+ We do not guarantee the exactly same results with the original implementations. (There might be slight numerical differences)
+ If you are not satisfied with the above remarks, please use the original code by the authors.

## Required Environments
+ Python 3
+ Chainer (v2 or v3)
+ OpenCV-Python
+ CUDA + CuDNN (for the GPU mode)

## Install
```
pip install chainer
pip install opencv-contrib-python==3.2.0.8
```

## Usage
After installing the required environments, do either of the following commands.
```
% For CPU mode
python demo.py

% For GPU mode
python demo.py -g 1

% For saving cost volume data
python demo.py -g 1 -v 1
```

## Reference
This code is intended to facilitate generation of cost volume data for input of our stereo method below. 
If you also find this work useful for your research, please cite our TPAMI paper.
```
@article{Taniai18,
  author    = {Tatsunori Taniai and
               Yasuyuki Matsushita and
               Yoichi Sato and
               Takeshi Naemura},
  title     = {{Continuous 3D Label Stereo Matching using Local Expansion Moves}},
  journal   = {{IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}},
  year      = {2018},
  volume    = {40},
  number    = {11},
  pages     = {2725--2739},
  doi       = {10.1109/TPAMI.2017.2766072},
}
```
See also: [[Project]](http://taniai.space/projects/stereo/)  [[GitHub]](https://github.com/t-taniai/LocalExpStereo)  [[Preprint]](https://arxiv.org/abs/1603.08328)
