# MC-MLP
In deep learning, Multi-Layer Perceptrons (MLPs) have once again garnered attention from researchers. This paper introduces MC-MLP, a general MLP-like backbone for computer vision that is composed of a series of fully-connected (FC) layers. In MC-MLP, we propose that the same semantic information has varying levels of difficulty in learning, depending on the coordinate frame of features. To address this, we perform an orthogonal transform on the feature information, equivalent to changing the coordinate frame of features. Through this design, MC-MLP is equipped with multi-coordinate frame receptive fields and the ability to learn information across different coordinate frames. Experiments demonstrate that MC-MLP outperforms most MLPs in image classification tasks, achieving better performance at the same parameter level. 
![image](https://user-images.githubusercontent.com/63572595/230283882-963083fc-c8cb-49d2-ae48-86bd92f7e17d.png)
|arch | Model     | Params     | GMAC     | Top-1(%)|
|--------| -------- | -------- | -------- |---------|
|CNN| ResNet-50 | 25.6 | 4.1 |75.1|
|Transfomer	|ViT|	22.1|	4.6|	66.45|
|CNN-ViT	|Conformer|	23.5	|5.2	|75.43|
|MLP|	MLP-Mixer|	28.0|	4.4|	62.16|
|MLP|ViP	|25.0	|8.5	|70.51|
|MLP|ResMLP|	30.0|	6.0|	66.40|
|MLP|AS-MLP	|28.0	|4.4|	65.16|
|MLP|gMLP	|24.5	|5.56|	64.80|
|MLP|MC-MLP(ours)|	25.8	|6.0|	78.64|
