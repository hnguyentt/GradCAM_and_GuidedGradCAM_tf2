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

![img](assets/illustrations/demo.gif)

## References:
* GradCAM paper: https://arxiv.org/abs/1610.02391
* GradCAM tutorial by `pyimagesearch`: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
* Keras GradCAM with Tensorflow 1.x backend:
    + https://github.com/jacobgil/keras-grad-cam
    + https://github.com/eclique/keras-gradcam