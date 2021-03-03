# HAINet
   This project provides the code and results for 'Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection', IEEE TIP 2021.
   
   
# Requirements
   python2.7
   
   pytorch 0.4.0
   
   Our code is implemented based on the environment settings of [CPD](https://github.com/wuzhe71/CPD). 


# Network Architecture
   ![Image](https://github.com/MathLee/HAINet/blob/main/Images/NetworkOverview.png)


# Testing
The code will come in soon.


# RGB-D SOD Results Trained with NJU2K and NLPR
   We provide [results](https://pan.baidu.com/s/1XkrwzNz9IDihz6S5gpfomg) (code: a2as) of our HAINet on 5 datasets (STEREO1000, NJU2K, DES, NLPR and SIP) and additional 2 datasets (SSD and LFSD).
   
   ![Image](https://github.com/MathLee/HAINet/blob/master/Images/Table1.png)
   
   
# RGB-D SOD Results Trained with NJU2K, NLPR and DUTLF-Depth
   We provide [results](https://pan.baidu.com/s/16CWNrzW7-b-0_hTSojirLg) (code: n35b) of our HAINet on 7 datasets (STEREO1000, NJU2K, DES, NLPR, SIP, DUTLF-Depth and [ReDWeb-S](https://github.com/nnizhang/SMAC)).
   
   ![Image](https://github.com/MathLee/HAINet/blob/master/Images/Table2.png)
   
   
# RGB-T SOD Results
   We apply our HAINet to [RGB-T SOD](https://github.com/lz118/RGBT-Salient-Object-Detection), and provide [results](https://pan.baidu.com/s/1A-lCEmitQUtAeqhPh4qpVQ) (code: s82s) of our HAINet on VT821 dataset trained with VT1000 dataset.
   
   ![Image](https://github.com/MathLee/HAINet/blob/master/Images/Table3.png)
   
   
# Evaluation tool
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
                volume = {},
                pages = {},
                doi = {10.1109/TIP.2020.2976689},}
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
