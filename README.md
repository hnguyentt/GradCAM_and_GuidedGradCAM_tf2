# Demo GradCAM & Guided GradCAM
> On [Dogs vs. Cat data](https://www.kaggle.com/c/dogs-vs-cats)

> Architecture: ResNet50 & ResNet50 + FC layers

An interactive demo for `GradCAM` and `Guided GradCAM`, implemented with Tensorflow 2.x

Detailed analysis and training notebook: https://www.kaggle.com/nguyenhoa/dog-cat-classifier-gradcam-with-tensorflow-2-0

## Prerequisite
* Python 3.6
* Required packages
```bash
bash requirements.txt
```

## Demo
Run file `Visualization.ipynb`

[img](assets/illustrations/demo.gif)
* Model: There are two trained models, which are
    * VanilaResNet50: Keep the same architecture of ResNet50, replace the output layer on ImageNet and re-train with Dog vs. Cat data.
    * ResNet50PlusFC: Add 2 fully connected layers between `Average Pooling` layer and output layer and train on Dog vs. Cat data.
* Image: There are some available sample images in `assets/samples`, if you want to run your own ones, put them in this folder to be displayed on the dropdown list.
* Class: This will be the class for GradCAM and Guided GradCAM visualization.

## References:
* GradCAM paper: https://arxiv.org/abs/1610.02391
* GradCAM tutorial by `pyimagesearch`: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
* Keras GradCAM with Tensorflow 1.x backend:
    + https://github.com/jacobgil/keras-grad-cam
    + https://github.com/eclique/keras-gradcam