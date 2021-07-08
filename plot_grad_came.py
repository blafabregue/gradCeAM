"""
Main script to launch experiments.
See parse_arguments() function for details on arguments

Author:
Baptiste Lafabregue 2021.06.01
"""

import numpy as np
import argparse
import csv
import types

import tensorflow as tf
import tensorflow_datasets as tfds
import imutils
import cv2

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.collections import LineCollection
from matplotlib._enums import CapStyle
from sklearn.cluster import KMeans
import cairosvg

import utils
from GradCam import GradCAM
from dnn.AutoEncoder_2D import CAE
from dnn.Resnet_1D import ResnetAE

DATA_2D = 'CAE-2D'
DATA_1D = 'ResNet-1D'


def export_plot(filenamebase):
    # save fil first to svg, then convert it to pdf to fix pdf saving with LineCollection
    pdf = "{}.pdf".format(filenamebase)
    svg = "{}.svg".format(filenamebase)
    plt.savefig(svg)
    cairosvg.svg2pdf(url=svg, write_to=pdf)


def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # apply the supplied color map to the heatmap and then
    # overlay the heatmap on the input image
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    # return a 2-tuple of the color mapped heatmap and the output,
    # overlaid image
    return heatmap, output


def plot_time_series(ts_array, label_true, path, ts_len, gradcams):
    nb_clusters = len(ts_array)
    ts_array = np.array(ts_array)
    min_val = np.min(ts_array)
    max_val = np.max(ts_array)

    # fig, axs = plt.subplots(nb_samples, nb_clusters, sharey=True, sharex=True,
    #                         figsize=(8 * nb_clusters, 3 * nb_samples))

    for c in range(nb_clusters):
        for i, (ts, label) in enumerate(zip(ts_array[c], label_true[c])):
            for j, grad in enumerate(gradcams):
                fig, axs = plt.subplots()
                # Compute the heatmaps and guided backpropagations
                heatmap, guided_backprop, decomposition, _ = grad.compute_heatmap_batch(np.array([ts]))
                heatmap, guided_backprop, decomposition = heatmap[0], guided_backprop[0], decomposition[0]
                norm = plt.Normalize(heatmap.min(), heatmap.max())

                # plot the time series with the heatmaps
                cmap = 'Spectral_r'
                points = np.array([np.arange(ts_len), ts[:, 0]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lines = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
                lines.set_array(heatmap)
                axs.add_collection(lines)
                axs.set_xlim(0, ts_len)
                axs.set_ylim(min_val, max_val)
                axs.get_xaxis().set_visible(False)
                axs.get_yaxis().set_visible(False)
                fig.tight_layout()

                utils.create_directory(path + str(j))
                plt.savefig(path + str(j) + '/ts_' + str(c) + '_' + str(i) + '.png')
                export_plot(path + str(j) + '/ts_' + str(c) + '_' + str(i))
                plt.close()

                # plot the guided backpropagations
                fig, axs = plt.subplots()
                axs.plot(np.arange(len(guided_backprop)), guided_backprop, linewidth=4)
                axs.set_xlim(0, ts_len)
                axs.set_ylim(min_val, max_val)
                axs.get_xaxis().set_visible(False)
                axs.get_xaxis().set_visible(False)
                axs.get_yaxis().set_visible(False)
                fig.tight_layout()

                utils.create_directory(path + str(j))
                plt.savefig(path + str(j) + '/guidedbp_' + str(c) + '_' + str(i) + '.png')
                export_plot(path + str(j) + '/guidedbp_' + str(c) + '_' + str(i))
                plt.close()


def plot_image(image_array, label_true, path, gradcams, plot_decomposition=False):
    nb_clusters = len(image_array)
    for c in range(nb_clusters):
        for i, (image, label) in enumerate(zip(image_array[c], label_true[c])):
            for j, grad in enumerate(gradcams):
                # Compute the heatmaps and guided backpropagations
                heatmap, guided_backprop, decomposition, _ = grad.compute_heatmap_batch(np.array([image]))
                heatmap, guided_backprop, decomposition = heatmap[0], guided_backprop[0], decomposition[0]

                # plot the time series with the heatmaps
                image_expand = image
                if image_expand.shape[-1] == 1:
                    image_expand = np.repeat(image_expand, 3, axis=-1)
                else:
                    # fix problem with stl10
                    image_expand = image_expand[:, :, [2, 1, 0]]
                image_expand = cv2.resize(image_expand, (280, 280))
                heatmap = cv2.resize(heatmap, (image_expand.shape[1], image_expand.shape[0]))
                (heatmap, output) = overlay_heatmap(heatmap, image_expand, alpha=0.5)
                cv2.rectangle(output, (0, 0), (30, 20), (0, 0, 0), -1)
                cv2.putText(output, str(label), (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                # display the original image and resulting heatmap and output image
                # to our screen
                output = np.vstack([image_expand, heatmap, output])
                output = imutils.resize(output, height=840)
                utils.create_directory(path + str(j))
                cv2.imwrite(path + str(j) + '/image_' + str(c) + '_' + str(i) + '.png', output)

                # plot the guided backpropagations
                guided_backprop = cv2.resize(guided_backprop, (image_expand.shape[1], image_expand.shape[0]))
                if len(guided_backprop.shape) < 3:
                    guided_backprop = np.expand_dims(guided_backprop, axis=-1)
                    guided_backprop = np.repeat(guided_backprop, 3, axis=-1)
                (_, output) = overlay_heatmap(guided_backprop, image_expand, alpha=0.5)
                cv2.rectangle(output, (0, 0), (30, 20), (0, 0, 0), -1)
                cv2.putText(output, str(label), (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                # display the original image and resulting heatmap and output image
                # to our screen
                output = np.vstack([image_expand, guided_backprop, output])
                output = imutils.resize(output, height=840)
                utils.create_directory(path + str(j))
                cv2.imwrite(path + str(j) + '/guidedbp_' + str(c) + '_' + str(i) + '.png', output)

                if plot_decomposition:
                    for d, heatmap in enumerate(decomposition):
                        heatmap = cv2.resize(heatmap, (image_expand.shape[1], image_expand.shape[0]))
                        (heatmap, output) = overlay_heatmap(heatmap, image_expand, alpha=0.5)
                        cv2.rectangle(output, (0, 0), (30, 20), (0, 0, 0), -1)
                        cv2.putText(output, str(label), (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 2)
                        # display the original image and resulting heatmap and output image
                        # to our screen
                        output = np.vstack([image_expand, heatmap, output])
                        output = imutils.resize(output, height=840)
                        utils.create_directory(path + str(j))
                        cv2.imwrite(
                            path + str(j) + '/image_' + str(c) + '_' + str(i) + '_decomposition' + str(d) + '.png',
                            output)


def plot_trainer_results(framework_name, features, x, y_true, y_pred, centers, nb_samples, path,
                         gradcams, center_reconstructions, x_bonus=None):
    data_array_best = []
    label_true_best = []
    data_array_worst = []
    label_true_worst = []
    data_array_wrong = []
    label_true_wrong = []
    clusters = np.unique(y_pred)

    # compute best assignment between clusters and classes
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    _, col = linear_sum_assignment(w.max() - w)
    match = np.zeros(len(clusters))
    for i in range(len(clusters)):
        match[i] = col[clusters[i]]
    match = match.astype(int)

    # get the top best images/time series that match/mismatch each cluster
    for clus in clusters:
        # compute distances
        rep = features[y_pred == clus]
        labels = y_true[y_pred == clus]
        raw_x = x[y_pred == clus]
        dist = np.array(list(map(lambda l: np.sum(np.power((l - centers[clus]), 2)), rep)))
        index_sorted = np.argsort(dist)

        # get best matches
        to_sample = np.min((nb_samples, len(index_sorted)))
        indexes_best = index_sorted[:to_sample]
        if len(indexes_best) < nb_samples:
            for k in range(nb_samples - len(indexes_best)):
                indexes_best = np.concatenate((indexes_best, np.array([indexes_best[0]])))
        data_array_best.append(raw_x[indexes_best])
        label_true_best.append(labels[indexes_best])

        # get worst matches
        from_sample = np.max((0, len(index_sorted) - nb_samples))
        indexes_worst = index_sorted[from_sample:]
        if len(indexes_worst) < nb_samples:
            for k in range(nb_samples - len(indexes_worst)):
                indexes_worst = np.concatenate((indexes_worst, np.array([indexes_worst[0]])))
        data_array_worst.append(raw_x[indexes_worst])
        label_true_worst.append(labels[indexes_worst])

        # get wrongly classified instances
        indexes_wrong = np.arange(len(labels))[labels != match[clus]]
        to_sample = np.min((nb_samples, len(indexes_wrong)))
        indexes_wrong = indexes_wrong[:to_sample]
        data_array_wrong.append(np.array(raw_x[indexes_wrong]))
        label_true_wrong.append(labels[indexes_wrong])

    centers_labels = np.arange(len(clusters))
    centers_labels = np.expand_dims(centers_labels, axis=1)

    # choose the good plotting option depending of the data type
    if framework_name == DATA_1D:
        ts_len = len(x[0])
        plot_time_series(np.expand_dims(center_reconstructions, axis=1), centers_labels,
                         path + '_centers', ts_len, gradcams)
        plot_time_series(data_array_best, label_true_best, path + '_best', ts_len, gradcams)
        plot_time_series(data_array_worst, label_true_worst, path + '_worst', ts_len, gradcams)
        # plot_time_series(data_array_wrong, label_true_worst, path + '_wrong', ts_len, gradcams)
        if x_bonus is not None:
            plot_time_series(x_bonus, np.zeros(len(x_bonus)), path + '_bonus', ts_len, gradcams)
    else:
        center_reconstructions = (center_reconstructions * 255).astype("uint8")
        plot_image(np.expand_dims(center_reconstructions, axis=1), centers_labels, path + '_centers', gradcams)
        data_array_best = (np.array(data_array_best) * 255).astype("uint8")
        plot_image(data_array_best, label_true_best, path + '_best', gradcams)
        data_array_worst = (np.array(data_array_worst) * 255).astype("uint8")
        plot_image(data_array_worst, label_true_worst, path + '_worst', gradcams)
        if x_bonus is not None:
            x_bonus = (np.array([x_bonus]) * 255).astype("uint8")
            plot_image(x_bonus, np.zeros(x_bonus.shape[:2]).astype("uint8"), path + '_bonus', gradcams)


def launch_plots(save_path, model, framework_name, dataset_name,
                 x_train, y_train, x_test, y_test, nb_samples, x_bonus=None,
                 y_pred_test=None, centers=None, features_test=None, kmeans=None):
    if len(centers.shape) > 2:
        centers = np.reshape(centers, (centers.shape[0] * centers.shape[1], centers.shape[2]))

    utils.create_directory(save_path)
    save_path += dataset_name + '/grad_'

    center_reconstructions = model.reconstruct_features(centers, already_encoded=True)

    gradcams = []
    for i, c in enumerate(centers):
        gradcams.append(GradCAM(model.encoder, i, centers, conv_dim=len(x_train.shape) - 2))

    if kmeans is None:
        kmeans = KMeans(n_clusters=len(centers), n_init=1, init=centers, max_iter=1)
        y_pred_test = kmeans.fit_predict(features_test)

    plot_trainer_results(framework_name, features_test, x_test, y_test, y_pred_test, centers, nb_samples,
                         save_path + 'test', gradcams, center_reconstructions, x_bonus=x_bonus)

    # evaluate gradcam dicrimination
    print('Cluster ratio: from,to')
    for i in range(len(centers)):
        # first compute the initial cluster ratio
        unique, counts = np.unique(y_pred_test, return_counts=True)
        count_per_cluster = dict(zip(unique, counts))
        initial_ratio = count_per_cluster[i] / len(x_test)

        _, _, _, guided_conv_ouput = gradcams[i].compute_heatmap_batch(x_test, compute_backprop=False,
                                                                       compute_decomposition=False)
        x_guided = gradcams[i].compute_prediction_from_target_layer(guided_conv_ouput)
        # compute the new ratio
        guided_pred = kmeans.predict(x_guided)
        unique, counts = np.unique(guided_pred, return_counts=True)
        count_per_cluster = dict(zip(unique, counts))
        try:
            guided_ratio = count_per_cluster[i] / len(x_test)
        except:
            guided_ratio = 0.0

        print(str(initial_ratio) + ',' + str(guided_ratio))


def preprocess_images(images):
    nb_features = 1
    if len(images.shape) > 3:
        nb_features = images.shape[3]
    images = images.reshape((images.shape[0], images.shape[1],
                             images.shape[2], nb_features)) / 255.
    return images.astype('float32')


def preprocess_images_black_and_white(images):
    nb_features = 1
    if len(images.shape) > 3:
        nb_features = images.shape[3]
    images = images.reshape((images.shape[0], images.shape[1],
                             images.shape[2], nb_features)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='d', required=True,
                        help='dataset name')
    parser.add_argument('--weights', default=False, action="store_true",
                        help='Flag to load existing autoencoder weights')

    return parser.parse_args()


def main(dataset, load_ae_weights=False):
    print('Launch on : ' + dataset)
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    x_bonus = None
    strides = [2, 2]
    filters = [32, 64]
    kernels = [3, 5]
    if dataset in ['MNIST', 'CIFAR', 'stl10', 'FMNIST']:
        framework_name = DATA_2D
        if dataset == 'MNIST':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = preprocess_images_black_and_white(x_train)
            x_test = preprocess_images_black_and_white(x_test)
        if dataset == 'FMNIST':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train = preprocess_images(x_train)
            x_test = preprocess_images(x_test)
        if dataset == 'CIFAR':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]
            x_train = preprocess_images(x_train)
            x_test = preprocess_images(x_test)
        if dataset == 'stl10':
            x_train, y_train = tfds.as_numpy(tfds.load(
                dataset,
                split='train',
                batch_size=-1,
                as_supervised=True,
            ))
            x_test, y_test = tfds.as_numpy(tfds.load(
                dataset,
                split='test',
                batch_size=-1,
                as_supervised=True,
            ))

            x_train = preprocess_images(x_train)
            x_test = preprocess_images(x_test)
            strides = [2, 2, 2]
            filters = [32, 64, 64]
            kernels = [3, 5, 5]
    else:
        framework_name = DATA_1D
        train_dict = utils.read_dataset('.', 'UCRArchive_2018', dataset, True)
        x_train = train_dict[dataset][0]
        y_train = train_dict[dataset][1]

        test_dict = utils.read_dataset('.', 'UCRArchive_2018', dataset, False)
        x_test = test_dict[dataset][0]
        y_test = test_dict[dataset][1]

    encoder_directory = './models/' + framework_name + '/' + dataset + '/'
    encoder_directory = utils.create_directory(encoder_directory)

    ae_weights = None
    if load_ae_weights:
        ae_weights = encoder_directory

    if framework_name == DATA_1D:
        model = ResnetAE(x_train, y_train, latent_dim=10)
        # model.encoder_loss = MSELoss(autoencoder=AutoencoderModel(model.encoder, model.decoder))
        if ae_weights is not None:
            model.load_weights(ae_weights)
        else:
            with open('./results/' + dataset + '.csv', 'w', newline='') as csvfile:
                logger = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                logger.writerow(['nmi', 'epoch', 'loss'])
                model.train(x_train, y_train, logger, x_test, y_test, epochs=50)
                model.save_weights(encoder_directory)
    else:
        model = CAE(x_train, y_train, latent_dim=10,
                    strides=strides, filters=filters, kernels=kernels)
        if ae_weights is not None:
            model.load_weights(ae_weights)
            print('weights successfully loaded')
        else:
            utils.create_directory('./results/')
            with open('./results/' + dataset + '.csv', 'w', newline='') as csvfile:
                logger = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                logger.writerow(['nmi', 'epoch', 'loss'])
                model.train(x_train, y_train, x_test, y_test, logger, epochs=2)
            model.save_weights(encoder_directory)

    nb_cluster = len(np.unique(y_test))
    features_test = model.extract_features(x_test)
    kmeans = KMeans(n_clusters=nb_cluster, n_init=25)
    y_pred_test, centers = kmeans.fit_predict(features_test), kmeans.cluster_centers_

    launch_plots('./plots/', model, framework_name, dataset,
                 x_train, y_train, x_test, y_test, 5, x_bonus=x_bonus, y_pred_test=y_pred_test,
                 centers=centers, features_test=features_test, kmeans=kmeans)


if __name__ == '__main__':
    class GC(GraphicsContextBase):
        def __init__(self):
            super().__init__()
            # self._capstyle = 'round'
            self._capstyle = CapStyle.round


    def custom_new_gc(self):
        return GC()


    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)

    args = parse_arguments()

    tf.keras.backend.set_floatx('float64')

    main(args.dataset, load_ae_weights=args.weights)
