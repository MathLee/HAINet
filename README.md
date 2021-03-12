# HAINet
   This project provides the code and results for 'Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection', IEEE TIP 2021. [Paper link](https://ieeexplore.ieee.org/document/9371407).
 
 
# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/HAINet/blob/main/Images/NetworkOverview.png">
   </div>
   
   
# Requirements
   python2.7
   
   pytorch 0.4.0
   
   Our code is implemented based on the environment settings of [CPD](https://github.com/wuzhe71/CPD). 


# Usage

Modify the pathes of [VGG backbone](https://pan.baidu.com/s/1YQxKZ-y2C4EsqrgKNI7qrw) (code: ego5) and datasets, then run train_HAI.py or test_HAI.py


# Pre-trained model
[Trained with NJU2K and NLPR](https://pan.baidu.com/s/1h5TqkeE3HatcRWVm7HmYZg) (code: 4ntl)

[Trained with NJU2K, NLPR and DUTLF-Depth](https://pan.baidu.com/s/1VsRQXHU_F6uQ9Vi97b_dmQ) (code: t5lr)


# RGB-D SOD Results Trained with NJU2K and NLPR
   We provide [results](https://pan.baidu.com/s/1XkrwzNz9IDihz6S5gpfomg) (code: a2as) of our HAINet on 5 datasets (STEREO1000, NJU2K, DES, NLPR and SIP) and additional 2 datasets (SSD and LFSD).
   
   ![Image](https://github.com/MathLee/HAINet/blob/main/Images/Table1.png)
   
   
# RGB-D SOD Results Trained with NJU2K, NLPR and DUTLF-Depth
   We provide [results](https://pan.baidu.com/s/16CWNrzW7-b-0_hTSojirLg) (code: n35b) of our HAINet on 7 datasets (STEREO1000, NJU2K, DES, NLPR, SIP, DUTLF-Depth and [ReDWeb-S](https://github.com/nnizhang/SMAC)).
   
   ![Image](https://github.com/MathLee/HAINet/blob/main/Images/Table2.png)
   
   
# RGB-T SOD Results
   We apply our HAINet to [RGB-T SOD](https://github.com/lz118/RGBT-Salient-Object-Detection), and provide [results](https://pan.baidu.com/s/1A-lCEmitQUtAeqhPh4qpVQ) (code: s82s) of our HAINet on VT821 dataset trained with VT1000 dataset.
   
   <div align=center>
   <img src="https://github.com/MathLee/HAINet/blob/main/Images/Table3.png">
   </div>
   
   
# Evaluation Tool
   You can use the [evaluation tool](http://dpfan.net/d3netbenchmark/) to evaluate the above saliency maps.


# Related works on RGB-D SOD
   (**ECCV_2020_CMWNet**) [Cross-Modal Weighting Network for RGB-D Salient Object Detection](https://github.com/MathLee/CMWNet).
   
   (**TIP_2020_ICNet**) [ICNet: Information Conversion Network for RGB-D Based Salient Object Detection](https://github.com/MathLee/ICNet-for-RGBD-SOD).
   
   (**Survey**) [RGB-D Salient Object Detection: A Survey](https://github.com/taozh2017/RGBD-SODsurvey).
   
# Citation
        @ARTICLE{Li_2021_HAINet,
                author = {Gongyang Li and Zhi Liu and Minyu Chen and Zhen Bai and Weisi Lin and Haibin Ling},
                title = {Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection},
                journal = {IEEE Transactions on Image Processing},
                year = {2021},
                volume = {30},
                pages = {3528-3542},}
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
