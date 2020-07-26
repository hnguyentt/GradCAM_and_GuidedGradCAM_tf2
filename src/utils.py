# Utility functions
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np

SAMPLE_DIR = "./assets/samples"
TARGET_SIZE = (224,224)
DECODE = {0:"cat", 1:"dog"}


def preprocess(filename):
    im = img_to_array(load_img(os.path.join(SAMPLE_DIR,filename),target_size = TARGET_SIZE))
    x = np.expand_dims(im, axis=0)
    x = preprocess_input(x)

    return x


def predict(model, processed_im):
    preds = model.predict(processed_im)
    idx = preds.argmax()

    res = [DECODE[idx], preds.max()]

    return res