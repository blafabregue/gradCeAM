import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import csv

import utils


class CAE():
    """Convolutional autoencoder."""

    def __init__(self, x, y, latent_dim, batch_size=64,
                 strides=[2, 2], kernels=[3, 5], filters=[32, 64]):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.n_clusters = len(np.unique(y))
        i_encoder = tf.keras.layers.Input(shape=x.shape[1:])
        # h = tf.keras.layers.Dropout(0.1)(i_encoder)
        h = i_encoder
        for s, k, f in zip(strides, kernels, filters):
            h = tf.keras.layers.Conv2D(filters=f, kernel_size=k,
                                       strides=(s, s), activation='relu')(h)
        # h = tf.keras.layers.GlobalAveragePooling2D()(h)
        h = tf.keras.layers.Flatten()(h)
        # No activation
        h = tf.keras.layers.Dense(latent_dim)(h)
        self.encoder = tf.keras.Model(inputs=i_encoder, outputs=h)

        # compute upsample shape
        upsample_h = x.shape[1]
        upsample_w = x.shape[2]
        for s in strides:
            upsample_h = upsample_h // s
            upsample_w = upsample_w // s
        upsample_h = int(np.floor(upsample_h))
        upsample_w = int(np.floor(upsample_w))

        upsample_remain = []
        base_shape = [x.shape[1], x.shape[2]]

        for s in strides:
            remain_h = base_shape[0] / s % 1.0
            base_shape[0] = base_shape[0] // s
            remain_w = base_shape[1] / s % 1.0
            base_shape[1] = base_shape[1] // s
            upsample_remain.append(((int(np.ceil(remain_h * s)), 0),
                                    (int(np.ceil(remain_w * s)), 0)))

        i_decoder = tf.keras.layers.Input(shape=(latent_dim,))
        h = tf.keras.layers.Dense(units=upsample_h * upsample_w * 32, activation=tf.nn.relu)(i_decoder)
        h = tf.keras.layers.Reshape(target_shape=(upsample_h, upsample_w, 32))(h)
        for i in range(len(strides) - 1, -1, -1):
            h = tf.keras.layers.Conv2DTranspose(
                filters=filters[i], kernel_size=kernels[i], strides=strides[i], padding='same',
                activation='relu')(h)
            h = tf.keras.layers.ZeroPadding2D(upsample_remain[i])(h)
        h = tf.keras.layers.Conv2DTranspose(
            filters=x.shape[-1], kernel_size=3, strides=1, padding='same')(h)
        h = tf.keras.layers.Activation('sigmoid')(h)
        self.decoder = tf.keras.Model(inputs=i_decoder, outputs=h)

    def save_weights(self, weights_path):
        """
        Save model's weights
        :param weights_path: path to save weights to
        """
        self.encoder.save_weights(weights_path + '_encoder.tf')
        self.decoder.save_weights(weights_path + '_decoder.tf')

    def load_weights(self, weights_path):
        """
        Load model's weights
        :param weights_path: path to load weights from
        """
        self.encoder.load_weights(weights_path + '_encoder.tf')
        self.decoder.load_weights(weights_path + '_decoder.tf')

    def encode(self, x, predict=False):
        if predict:
            return self.encoder.predict(x)
        return self.encoder(x)

    def decode(self, z, predict=False):
        if predict:
            return self.decoder.predict(z)
        return self.decoder(z)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def reconstruct_features(self, x, already_encoded=False):
        if already_encoded:
            z = x
        else:
            z = self.encoder.predict(x)
        return self.decoder.predict(z)

    def log_stats(self, x, y, x_test, y_test, loss, epoch, log_writer, comment):
        """
        Log the intermediate result to a file
        :param x: train data
        :param y: train labels
        :param x_test: test data
        :param y_test: test labels
        :param loss: array of losses values
        :param epoch: current epoch
        :param log_writer: log file writer
        :param comment: comment to add to the log
        :return:
        """
        loss = np.round(loss, 5)
        logs = {'acc': 0, 'nmi': 0, 'ari': 0}
        logs_test = {'acc': 0, 'nmi': 0, 'ari': 0}

        if y is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            x_pred = self.extract_features(x)
            y_pred = kmeans.fit_predict(x_pred)

            logs['acc'] = np.round(utils.cluster_acc(y, y_pred), 5)
            logs['nmi'] = np.round(normalized_mutual_info_score(y, y_pred), 5)
            logs['ari'] = np.round(adjusted_rand_score(y, y_pred), 5)
            print('acc: ' + str(logs['acc']) + ' nmi: ' + str(logs['nmi']) + ' epoch: ' + str(epoch))

        if x_test is not None and y_test is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            x_pred_test = self.extract_features(x_test)
            y_pred_test = kmeans.fit_predict(x_pred_test)

            logs_test['acc'] = np.round(utils.cluster_acc(y_test, y_pred_test), 5)
            logs_test['nmi'] = np.round(normalized_mutual_info_score(y_test, y_pred_test), 5)
            logs_test['ari'] = np.round(adjusted_rand_score(y_test, y_pred_test), 5)

        log_dict = dict(iter=epoch, L=loss,
                        acc=logs['acc'], nmi=logs['nmi'], ari=logs['ari'],
                        acc_test=logs_test['acc'], nmi_test=logs_test['nmi'],
                        ari_test=logs_test['ari'], comment=comment)
        log_writer.writerow(log_dict)

    def train(self, x, y, x_test, y_test, logger, epochs=30):
        train_enc_loss = tf.keras.metrics.Mean(name='AutoEncoder train_loss')
        train_dataset = (tf.data.Dataset.from_tensor_slices(x)
                         .shuffle(len(x)).batch(self.batch_size))

        @tf.function
        def train_step(_x_batch, optimizer):
            with tf.GradientTape() as tape:
                encode = self.encode(_x_batch)
                decode = self.decode(encode)
                loss = self.loss_function(_x_batch, decode)
            gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
            train_enc_loss(loss)

        for epoch in range(1, epochs + 1):
            train_enc_loss.reset_states()
            for x_batch in train_dataset:
                train_step(x_batch, self.optimizer)

            # encode = self.encode(x, predict=True)
            # kmeans = KMeans(n_clusters=nb_cluster).fit(encode)
            # y_pred = kmeans.predict(encode)

            # nmi = normalized_mutual_info_score(y, y_pred)
            #
            # print('NMI = ' + str(nmi) + ' epoch = ' + str(epoch) + ' loss = ' + str(train_enc_loss.result()))
            # logger.writerow([nmi, epoch, train_enc_loss.result()])
            self.log_stats(x, y, x_test, y_test, train_enc_loss.result().numpy(),
                           epoch, logger, 'training')


def preprocess_images(images):
    nb_features = 1
    if len(images.shape) > 3:
        nb_features = images.shape[3]
    images = images.reshape((images.shape[0], images.shape[1],
                             images.shape[2], nb_features)) / 255.
    return images.astype('float32')


def preprocess_images_back_and_white(images):
    nb_features = 1
    if len(images.shape) > 3:
        nb_features = images.shape[3]
    images = images.reshape((images.shape[0], images.shape[1],
                             images.shape[2], nb_features)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # dataset_name = 'mnist'
    # strides = [2, 2]
    # filters = [32, 64]
    # kernels = [3, 5]
    # x_train = preprocess_images_black_and_white(x_train)
    # x_test = preprocess_images_black_and_white(x_test)

    dataset_name = 'stl10'
    strides = [2, 2, 2]
    filters = [32, 64, 64]
    kernels = [3, 5, 5]
    train_dict = utils.read_dataset('..', 'Image', dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]
    input_shape = x_train.shape[1:]
    nb_classes = np.shape(np.unique(y_train, return_counts=True)[1])[0]

    test_dict = utils.read_dataset('..', 'Image', dataset_name, False)
    x_test = test_dict[dataset_name][0]
    y_test = test_dict[dataset_name][1]

    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    # x_train, y_train = tfds.as_numpy(tfds.load(
    #     dataset_name,
    #     split='train',
    #     batch_size=-1,
    #     as_supervised=True,
    # ))
    # x_test, y_test = tfds.as_numpy(tfds.load(
    #     dataset_name,
    #     split='test',
    #     batch_size=-1,
    #     as_supervised=True,
    # ))
    #
    # utils.create_directory('data/' + dataset_name)
    # np.save('../data/' + dataset_name + '/x_train.npy', x_train)
    # np.save('../data/' + dataset_name + '/y_train.npy', y_train)
    # np.save('../data/' + dataset_name + '/x_test.npy', x_test)
    # np.save('../data/' + dataset_name + '/y_test.npy', y_test)
    # np.save('../data/' + dataset_name + '/x_train.npy', x_train)
    # np.save('../data/' + dataset_name + '/y_train.npy', y_train)
    # np.save('../data/' + dataset_name + '/x_test.npy', x_test)
    # np.save('../data/' + dataset_name + '/y_test.npy', y_test)

    nb_cluster = len(np.unique(y_test))
    latent_dim = 10

    framework_name = 'CAE-2D'
    encoder_directory = '../models/' + framework_name + '/' + dataset_name + '/'
    encoder_directory = utils.create_directory(encoder_directory)

    model = CAE(x_train, y_train, latent_dim, strides=strides, filters=filters, kernels=kernels)
    with open(encoder_directory + 'logs.csv', 'w', newline='') as csvfile:
        log_writer = csv.DictWriter(csvfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'comment',
                                                         'acc_test', 'nmi_test', 'ari_test'])
        log_writer.writeheader()
        model.train(x_train, y_train, log_writer, x_test, y_test)
    # model.train(x_train, y_train)
    model.save_weights(encoder_directory)

    encode = model.encode(x_test, predict=True)
    kmeans = KMeans(n_clusters=nb_cluster).fit(encode)
    y_pred = kmeans.predict(encode)

    nmi = normalized_mutual_info_score(y_test, y_pred)

    print('NMI = ' + str(nmi))
