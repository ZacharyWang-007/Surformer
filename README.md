# Surformer : an interpretable pattern-perceptive survival transformer for cancer survival prediction from histopathology whole slide images

This is the official pytorch implementation of Surformer [Surformer : an interpretable pattern-perceptive survival transformer for cancer survival prediction from histopathology whole slide images](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4423682). 


## Pipline
<div align="center">
  <img src="Figures/fig1.png">
 </div>
 
 ## Experiment Results on five TCGA tumor datasets
 <div align="center">
  <img src="Figures/Result1.png" width="400px"/>
 </div>
 
 <div align="center">
  <img src="Figures/Result2.png" width="400px"/>
 </div>
 
 ## Retrieve Comparison between [TransReID](https://github.com/damo-cv/TransReID) 
 <div align="center">
  <img src="Figures/Comparison.png" width="700px"/>
 </div>
 
 
 ## Requirements
 ### Installation
 Please refer to [TransReID](https://github.com/damo-cv/TransReID) 
 ### Dataset Preparation
 Please download Occluded-Duke dataset and [cropped patches](https://drive.google.com/file/d/1lYTBokHR8pkbjz_LPhZjh4ij8B-FI1LA/view?usp=sharing). Meanwhile place cropped patches into Occluded-Duke (just because of dataloader).
 ### Pretrained Model Preparison
 Please download pretrained [ViT backbone](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) in advance.
 
 ## Model training and testing
 before training and testing, please update config file accordingly.  Around 13G GPU memory is required. 
 ~~~~~~~~~~~~~~~~~~
   python train.py 
 ~~~~~~~~~~~~~~~~~~

## Citation

If you find this code useful for your research, please cite our paper

```
@article{wang2023surformer,
  title={Surformer: An Interpretable Pattern-Perceptive Survival Transformer for Cancer Survival Prediction from Histopathology Whole Slide Images},
  author={Wang, Zhikang and Gao, Qian and Yi, Xiao-Ping and Zhang, Xinyu and Zhang, Yiwen and Zhang, Daokun and Li{\`o}, Pietro and Bain, Christopher and Bassed, Richard and Li, Shanshan and others},
  journal={Available at SSRN 4423682},
  year={2023}
}

@article{wang2023targeting,
  title={Targeting tumor heterogeneity: multiplex-detection-based multiple instance learning for whole slide image classification},
  author={Wang, Zhikang and Bi, Yue and Pan, Tong and Wang, Xiaoyu and Bain, Chris and Bassed, Richard and Imoto, Seiya and Yao, Jianhua and Daly, Roger J and Song, Jiangning},
  journal={Bioinformatics},
  volume={39},
  number={3},
  pages={btad114},
  year={2023},
  publisher={Oxford University Press}
}
```

## Contact

If you have any question, please feel free to contact us. E-mail: [zhikang.wang@monash.edu](zhikang.wang@monash.edu) 

