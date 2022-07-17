Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification, TIP, 2022.
==
[Bobo Xi](https://scholar.google.com/citations?user=O4O-s4AAAAAJ&hl=zh-CN), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html),  [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Danfeng Hong](https://sites.google.com/view/danfeng-hong/home) and [Jocelyn Chanussot](https://jocelyn-chanussot.net/).
***
Code for the paper: Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification. (The paper will be early accessed soon!)

<div align=center><img src="/Image/frameworks.jpg" width="80%" height="80%"></div>
Fig. 1: The architecture of the proposed CMFSL for HSIC. Based on the class-covariance metric, the classification process is completed by the episode-based collaboratively meta-training of the source and target data sets, and the episode-based meta-test of the target data set. Notably, the embedding feature extractor comprises a new SPRM and a novel LXConvNet.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. And the websites to access the datasets are also provided. The used OCBS band selection method is referred to [https://github.com/tanmlh] (https://github.com/tanmlh).
2) Run "trainMetaDataProcess.py" to generate the meta-training data 
3) Run the 'CMFSL_UP_main.py' to reproduce the CMFSL results on [Pavia University](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene) data set.

We have successfully tested it on Ubuntu 16.04 with PyTorch 1.1.0. Below is the classification map with five shots of training samples from each class. 

<div align=center><p float="center">
<img src="/Image/false_color.jpg" height="300" width="200"/>
<img src="/Image/gt.jpg" height="300"width="280"/>
<img src="/Image/classification_map.jpg" height="300"width="200"/>
</p></div>
<div align=center>Fig. 2: The composite false-color image, groundtruth, and classification map of Pavia University dataset.</div>  

References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, J. Li, Y. Li, R. Song, D. Hong, J. Chanussot, “Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification, IEEE Transactions on Image Processing, pp. 1-14, 2022. 


Citation Details
--
BibTeX entry:
```
@ARTICLE{Xi_2022TIP_CMFSL,
  author={Xi, Bobo and Li, Jiaojiao and Li, Yunsong and Song, Rui and Hong, Danfeng and Chanussot, Jocelyn},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  }
```
 
Licensing
--
Copyright (C) 2022 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

