"""
Inspired from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

Author:
Baptiste Lafabregue 2021.06.01
"""

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2



def compute_features_weights(centroid, other_centroids):
    weights = other_centroids - centroid
    weights = np.absolute(weights)
    weights = np.sum(weights, axis=0)
    weights = weights / len(weights)
    final_weights = weights - np.min(weights)
    final_weights /= np.max(weights) - np.min(weights)

    return final_weights


def scale(ts, new_length):
    old_length = len(ts)  # source number of time steps
    return np.array([ts[int(old_length * r / new_length)] for r in range(new_length)])


def normalize(input, eps=1e-8):
    numer = input - np.min(input)
    denom = (input.max() - input.min()) + eps
    res = numer / denom
    return res


def smooth_values(array, amplitude=0.5, diameter=-1):
    ts_len = len(array)
    result = np.zeros(ts_len)

    if diameter < 0:
        diameter = int(np.ceil(ts_len / 20))
    for i in range(ts_len):
        factor = 0
        start_shift = i - diameter if i - diameter > 0 else 0
        end_shift = i + diameter + 1 if i + diameter + 1 < ts_len - 1 else ts_len

        for j in range(start_shift, i):
            result[i] += amplitude / (i - j) * array[j]
            factor += amplitude / (i - j)

        for j in range(i + 1, end_shift):
            result[i] += amplitude / (j - i) * array[j]
            factor += amplitude / (j - i)

        result[i] += array[i]

    return np.array(result)


class GradCAM:

    def __init__(self, model, cluster_id, centroids, conv_dim=1, layer_name=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.centroid = centroids[cluster_id]
        self.all_centroids = tf.cast(centroids, tf.float64)
        self.nb_centroids = len(centroids)
        self.features_weight = compute_features_weights(self.centroid, np.delete(centroids, cluster_id, axis=0))
        self.layer_name = layer_name
        self.conv_dim = conv_dim
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()
        self.grad_model = None
        self.backprop_model = None
        self.tmp_model = None

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 3D/4D output
            if len(layer.output_shape) == (self.conv_dim + 2):
                return layer.name
        # otherwise, we could not find a  layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find " + str(self.conv_dim + 2) + "D layer. Cannot apply GradCAM.")

    def compute_prediction_from_target_layer(self, input):
        found = False

        if self.tmp_model is None:
            i = tf.keras.layers.Input(shape=input.shape[1:])
            h = i
            for layer in self.model.layers:
                # first find the target layer
                if not found:
                    if layer.name == self.layer_name:
                        found = True
                else:
                    h = layer(h)
            self.tmp_model = Model(
                inputs=[i],
                outputs=[h])
        return self.tmp_model.predict(input)

    def compute_heatmap_batch(self, input, eps=1e-3, compute_backprop=True, compute_decomposition=True):
        # handle the cas where the batch is too big
        if len(input) > 256:
            heatmap, guided_backprop, heatmap_decomposed, cam_decomposed = [], [], [], []
            for sub_array in np.split(input, int(np.ceil(len(input) / 256))):
                h, gb, hd, cd = self.compute_heatmap_batch(sub_array, compute_backprop=compute_backprop,
                                                           compute_decomposition=compute_decomposition)
                heatmap.append(h)
                guided_backprop.append(gb)
                heatmap_decomposed.append(hd)
                cam_decomposed.append(cd)
            heatmap = np.concatenate(heatmap, axis=0)
            if compute_backprop:
                guided_backprop = np.concatenate(guided_backprop, axis=0)
            else:
                guided_backprop = None
            if compute_decomposition:
                heatmap_decomposed = np.concatenate(heatmap_decomposed, axis=0)
            else:
                heatmap_decomposed = None
            cam_decomposed = np.concatenate(cam_decomposed, axis=0)
            return heatmap, guided_backprop, heatmap_decomposed, cam_decomposed

        ########################################################
        # First we create models if they do not exist yet
        ########################################################
        @tf.custom_gradient
        def guided_relu(x):
            def grad(dy):
                return tf.cast(dy > 0, "float64") * tf.cast(x > 0, "float64") * dy

            return tf.nn.relu(x), grad

        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 3D layer in the network, and (3) the encoder latent dim
        if self.grad_model is None:
            self.grad_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output,
                         self.model.output])

        # do the same for the backprop model but also change the relu activation
        if self.backprop_model is None:
            self.backprop_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output,
                         self.model.output])
            layer_dict = [layer for layer in self.backprop_model.layers[1:] if hasattr(layer, 'activation')]
            for layer in layer_dict:
                if layer.activation == tf.keras.activations.relu:
                    layer.activation = guided_relu

        ########################################################
        # We can compute the heatmap
        ########################################################
        input_size = len(input)
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(input, tf.float64)

            conv_outputs, prediction = self.grad_model(inputs)

            diff_centroid = tf.math.abs(tf.math.subtract(self.centroid, prediction))
            prediction_reshaped = tf.expand_dims(prediction, axis=0)
            prediction_reshaped = tf.repeat(prediction_reshaped, self.nb_centroids, axis=0)
            other_centroids_reshaped = tf.expand_dims(self.all_centroids, axis=1)
            other_centroids_reshaped = tf.repeat(other_centroids_reshaped, input_size, axis=1)
            diff_other_centroids = tf.math.abs(tf.math.subtract(other_centroids_reshaped, prediction_reshaped))
            diff_min = tf.reduce_min(diff_other_centroids, axis=0)
            diff_max = tf.reduce_max(diff_other_centroids, axis=0)

            centroid_ranking = tf.math.subtract(diff_centroid, diff_min)
            ranking_spread = tf.math.subtract(diff_max, diff_min)
            # ranking_spread = tf.math.add(ranking_spread, eps)
            centroid_ranking = tf.math.add(centroid_ranking, eps)
            centroid_ranking = tf.math.divide(ranking_spread, centroid_ranking)
            centroid_ranking = tf.math.multiply(tf.math.abs(centroid_ranking), centroid_ranking)

            # the loss is composed of the weight of each feature in the centroid choice
            # and the ranking of the current centroid per feature (is it the closest to the prediction or not)
            loss = tf.math.multiply(self.features_weight, centroid_ranking)
            loss = tf.math.multiply(loss, tf.math.abs(prediction))

        # use automatic differentiation to compute the gradients
        guided_grads = tape.gradient(loss, conv_outputs)

        # compute the guided gradients
        conv_outputs = conv_outputs.numpy()
        guided_grads = guided_grads.numpy()

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        axis = tuple(np.arange(1, self.conv_dim + 1))
        weights = np.sum(guided_grads, axis=axis)
        weights = weights * 2
        for i in range(self.conv_dim):
            weights = np.expand_dims(weights, axis=1)
        cam_decomposed = conv_outputs * weights
        cam = np.sum(tf.math.multiply(weights, conv_outputs), axis=-1)

        # rescale the heatmaps to fit the input
        if self.conv_dim == 1:
            heatmap = np.array([scale(c, input.shape[1]) for c in cam])
        else:
            heatmap = np.array([cv2.resize(c, (input.shape[2], input.shape[1])) for c in cam])

        # normalize the heatmaps
        saved_heatmap = heatmap = np.array([normalize(h) for h in heatmap])
        if self.conv_dim == 1:
            heatmap = np.array([smooth_values(h) for h in heatmap])
        else:
            heatmap = (heatmap * 255).astype("uint8")

        ########################################################
        # Compute decomposition if needed
        ########################################################
        heatmap_decomposed = []
        if compute_decomposition:
            for current_cam in cam_decomposed:
                current_decomposition = []
                for c in np.split(current_cam, current_cam.shape[-1], axis=-1):
                    c = np.squeeze(c)
                    if self.conv_dim == 1:
                        hm = scale(c, input.shape[1])
                    else:
                        hm = cv2.resize(c, (input.shape[2], input.shape[1]))

                    hm = normalize(hm)
                    if self.conv_dim == 1:
                        hm = smooth_values(hm)
                    else:
                        hm = (hm * 255).astype("uint8")
                    current_decomposition.append(hm)
                heatmap_decomposed.append(current_decomposition)
        heatmap_decomposed = np.array(heatmap_decomposed)

        ########################################################
        # Compute guided-backpropagation if needed
        # it needs to be computed separately because it rely on a modified model
        ########################################################
        guided_backprop = None
        if compute_backprop:
            with tf.GradientTape() as tape:
                inputs = tf.cast(input, tf.float64)
                tape.watch(inputs)

                _, prediction = self.backprop_model(inputs)

                diff_centroid = tf.math.subtract(self.centroid, prediction)
                prediction_reshaped = tf.expand_dims(prediction, axis=0)
                prediction_reshaped = tf.repeat(prediction_reshaped, self.nb_centroids, axis=0)
                other_centroids_reshaped = tf.expand_dims(self.all_centroids, axis=1)
                other_centroids_reshaped = tf.repeat(other_centroids_reshaped, input_size, axis=1)
                diff_other_centroids = tf.math.subtract(other_centroids_reshaped, prediction_reshaped)
                diff_min = tf.reduce_min(diff_other_centroids, axis=0)
                diff_max = tf.reduce_max(diff_other_centroids, axis=0)

                centroid_ranking = tf.math.subtract(diff_centroid, diff_min)
                ranking_spread = tf.math.subtract(diff_max, diff_min)
                centroid_ranking = tf.math.divide(centroid_ranking, ranking_spread)
                centroid_ranking = tf.math.multiply(tf.math.abs(centroid_ranking), centroid_ranking)

                loss = tf.math.multiply(self.features_weight, centroid_ranking)
                loss = tf.math.multiply(loss, prediction)

            # use automatic differentiation to compute the gradients
            grads = tape.gradient(loss, inputs)
            guided_backprop = grads.numpy()

            saved_heatmap = np.repeat(np.expand_dims(saved_heatmap, axis=-1), guided_backprop.shape[-1], axis=-1)
            guided_backprop = np.multiply(guided_backprop, saved_heatmap)
            guided_backprop = np.array([normalize(gb) for gb in guided_backprop])
            if self.conv_dim == 2:
                guided_backprop = (guided_backprop * 255).astype("uint8")

        return heatmap, guided_backprop, heatmap_decomposed, cam_decomposed
