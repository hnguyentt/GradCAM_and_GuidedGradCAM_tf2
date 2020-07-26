import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops

# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0
class GuidedBackprop:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_model(self):
        return tf.keras.models.clone_model(self.model)

    def build_guided_model(self):
        """Function returning modified model.

        Changes gradient function for all ReLu activations
        according to Guided Backpropagation.
        """
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                       tf.cast(op.inputs[0] > 0., dtype)

        g = tf.compat.v1.get_default_graph()
        with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
            new_model = self.build_model()

        return new_model

    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        guided_model = self.build_guided_model()
        gbModel = Model(
            inputs=[guided_model.input],
            outputs=[guided_model.get_layer(self.layerName).output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #     if K.image_dim_ordering() == 'th':
    #         x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x