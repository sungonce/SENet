# Revisiting Self-Similarity: Structural Embedding for Image Retrieval 
> Official Pytorch Implementation of the paper "[**Revisiting Self-Similarity: Structural Embedding for Image Retrieval**](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Revisiting_Self-Similarity_Structural_Embedding_for_Image_Retrieval_CVPR_2023_paper.html)"<br>
> accept to **CVPR 2023** <br>
> by Seongwon Lee, Suhyeon Lee, Hongje Seong, and Euntai Kim<br>
> Yonsei University
> 

### Introduction

<p align="middle">
    <img src="assets/senet.jpg">
</p>

## ➡️ Guide to Our Code

### Data preparation
#### Download [ROxford5k and RParis6k](https://github.com/filipradenovic/revisitop). Unzip the files and make the directory structures as follows.


```
revisitiop
 └ data
   └ datasets
     └ roxford5k
       └ gnd_roxford5k.pkl
       └ jpg
         └ ...
     └ rparis6k
       └ gnd_rparis6k.pkl
       └ jpg
         └ ...
```

### Pretrained models
You can download our pretrained models from [Google Drive](https://drive.google.com/drive/folders/1zKk54aKNdBH8sXB9x4J-O-n275XCFbX3?usp=sharing).

### Testing
For ResNet-50 model, run the command
```bash
python test.py SENET.RESNET_DEPTH 50 TEST.WEIGHTS <path-to-R50-pretrained-model> TEST.DATA_DIR <path_to_revisitop>/data/datasets
```

and for ResNet-101 model, run the command
```bash
python test.py SENET.RESNET_DEPTH 101 TEST.WEIGHTS <path-to-R101-pretrained-model> TEST.DATA_DIR <path_to_revisitop>/data/datasets
```

## 🙏 Acknowledgments
Our pytorch implementation is derived from [Revisiting Oxford and Paris](https://github.com/filipradenovic/revisitop) and [DELG-pytorch](https://github.com/feymanpriv/DELG). We thank for these great works and repos.

## ✏️ Citation
If you find our paper useful in your research, please cite us using the following entry:
````BibTeX
@InProceedings{lee2023senet, 
    author    = {Lee, Seongwon and Lee, Suhyeon and Seong, Hongje and Kim, Euntai},
    title     = {Revisiting Self-Similarity: Structural Embedding for Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23412-23421}
}
````