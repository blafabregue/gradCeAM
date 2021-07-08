import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Activation, Dropout, add
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import utils


class MSELoss:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.loss = tf.keras.losses.MeanSquaredError()

    def compute_loss(self, batch, noisy_batch=None, training=True):
        if noisy_batch is None:
            noisy_batch = batch
        encoding = self.encoder(noisy_batch, training=training)
        decoding = self.decoder(encoding, training=training)
        return self.loss(batch, decoding)


class ResnetAE:
    """Resnet autoencoder."""

    def __init__(self,
                 x, y,
                 encoder_loss=None,
                 filters=[64, 128, 128],
                 kernels=[8, 5, 3],
                 latent_dim=10,
                 batch_size=10,
                 activation='relu',
                 dropout_rate=0
                 ):
        self.n_clusters = len(np.unique(y))
        self.filters = np.array(filters)
        self.kernels = np.array(kernels)
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.encoder_loss = encoder_loss
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam()

        self.encoder = self._create_encoder(x)
        self.decoder = self._create_decoder(x)
        if self.encoder_loss is None:
            self.encoder_loss = MSELoss(self.encoder, self.decoder)

    def _create_res_block(self, i, n_features):
        conv_x = Conv1D(filters=n_features, kernel_size=int(self.kernels[0]), padding='same')(i)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_features, kernel_size=int(self.kernels[1]), padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_features, kernel_size=int(self.kernels[2]), padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_features, kernel_size=1, padding='same')(i)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_1 = add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        return output_block_1

    def _create_encoder(self, x):
        i = Input(shape=x.shape[1:])
        h = i

        for n_features in self.filters:
            h = self._create_res_block(h, n_features)
            if self.dropout_rate > 0:
                h = Dropout(self.dropout_rate)(h)

        h = GlobalAveragePooling1D()(h)
        h = Dense(self.latent_dim)(h)

        return Model(inputs=i, outputs=h)

    def _create_decoder(self, x):
        out_channels = 1

        for s in x.shape[1:]:
            out_channels *= s
        i = Input(shape=(self.latent_dim,))
        h = Dense(out_channels)(i)
        h = Reshape((x.shape[1:]))(h)
        for n_features in np.flip(self.filters):
            h = self._create_res_block(h, n_features)
        h = Conv1D(filters=x.shape[-1], kernel_size=int(self.kernels[-1]), padding='same', strides=1)(h)

        return Model(inputs=i, outputs=h)

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

    def train(self, x, y, logger, x_test, y_test, epochs=200):
        train_enc_loss = tf.keras.metrics.Mean(name='AutoEncoder train_loss')

        @tf.function
        def train_step(x_batch, noisy_batch):
            trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            with tf.GradientTape() as tape:
                loss = self.encoder_loss.compute_loss(x_batch, noisy_batch=noisy_batch)
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

            train_enc_loss(loss)

        for epoch in range(1, epochs + 1):
            train_enc_loss.reset_states()
            data = tf.data.Dataset.from_tensor_slices(x)
            indices = tf.data.Dataset.range(len(x))
            train_ds = tf.data.Dataset.zip((data, indices))
            train_ds = train_ds.shuffle(x.shape[0], reshuffle_each_iteration=True)
            train_ds = train_ds.batch(self.batch_size).as_numpy_iterator()
            for batch, idx in train_ds:
                self.encoder_loss.batch_idx = idx
                train_step(batch, batch)
            self.log_stats(x, y, x_test, y_test, train_enc_loss.result().numpy(),
                           epoch, logger, 'training')
