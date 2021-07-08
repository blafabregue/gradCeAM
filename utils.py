"""
tools used for the experiments

Based on Hassan Fawaz implementation https://github.com/hfawaz/dl-4-tsc

Author:
Baptiste Lafabregue 2021.06.01
"""

import numpy as np
import os
from sktime.datasets.base import load_UCR_UEA_dataset


def read_dataset(root_dir, archive_name, dataset_name, is_train=True):
    datasets_dict = {}

    if is_train:
        type = 'train'
    else:
        type = 'test'

    file_name = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
    x = np.load(file_name + 'x_' + type + '.npy', allow_pickle=True)
    y = np.load(file_name + 'y_' + type + '.npy', allow_pickle=True)
    y = y.astype(int)
    if len(x.shape) == 2:
        x = np.reshape(x, (x.shape[0], -1, 1))
    #     x = np.reshape(x, (x.shape[0], 1, -1))
    # else:
    #     x = np.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))

    datasets_dict[dataset_name] = (x.copy(), y.copy())
    datasets_dict['k'] = len(np.unique(y))

    return datasets_dict


def create_output_path(root_dir, itr, framework_name, dataset_name, type='ae_weights'):
    dir = root_dir + '/' + type + '/' + str(itr) + '/' + framework_name + '/' + dataset_name + '/'
    create_directory(dir)
    return dir


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return directory_path
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(w.max() - w)
    sum_i = sum([w[r[i], c[i]] for i in range(len(r))])
    return sum_i * 1.0 / y_pred.size


def transform_to_same_length(x, max_length):
    n = len(x)
    # n_var = x[0].shape[1]
    # pad = (0 for _ in range(n_var))

    # only use zero padding to follow univariate UCR method
    for i in range(n):
        if len(x[i].shape) == 1:
            x[i] = np.reshape(x[i], (1, x[i].shape[0]))
        x[i] = np.pad(x[i], ((0, max_length - x[i].shape[0]), (0, 0)), 'constant')

    # # the new set in ucr form np array
    # ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)
    #
    # # loop through each time series
    # for i in range(n):
    #     mts = x[i]
    #     curr_length = mts.shape[1]
    #     idx = np.array(range(curr_length))
    #     idx_new = np.linspace(0, idx.max(), max_length)
    #     for j in range(n_var):
    #         ts = mts[j]
    #         # linear interpolation
    #         new_ts = ts + idx_new
    #         ucr_x[i, :, j] = new_ts
    #
    # return ucr_x
    return x


def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = len(x_train)
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[0])

    n = len(x_test)
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[0])

    return func_length


def align(x, max_length):
    for j in range(len(x)):
        x[j] = np.pad(x[j], (0, max_length-len(x[j])), constant_values=(0, 0))


def transform_sktime_to_npy_format(mts_root_dir, mts_out_dir):
    dataset_files = [name for name in os.listdir(mts_root_dir)]
    # dataset_files = ['CharacterTrajectories']

    for dataset_name in dataset_files:
        out_dir = mts_out_dir + dataset_name + '/'
        create_directory(out_dir)

        x_train_df, y_train = load_UCR_UEA_dataset(
            mts_root_dir + dataset_name + '/' + dataset_name + '_TRAIN.ts')
        x_test_df, y_test = load_UCR_UEA_dataset(mts_root_dir + dataset_name + '/' + dataset_name + '_TEST.ts')

        try:
            # ensure to handle string of floats
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except:
            unique = np.unique(y_train)
            unique.sort()
            for i, val in enumerate(unique):
                y_train = np.where(y_train == val, i, y_train)
                y_test = np.where(y_test == val, i, y_test)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        dims = x_train_df.columns
        x_train = []
        for i in range(x_train_df[dims[0]].size):
            x_channels = []
            t = -1
            realign = False
            for d in dims:
                x_channels.append(x_train_df.loc[i][d].values)
                new_t = len(x_channels[-1])
                if t < 0:
                    t = new_t
                elif t != new_t:
                    t = max((t, new_t))
                    realign = True
            if realign:
                align(x_channels, t)

            x_train.append(np.array(x_channels).T)

        x_test = []
        for i in range(x_test_df[dims[0]].size):
            x_channels = []
            for d in dims:
                x_channels.append(x_test_df.loc[i][d].values)
            x_test.append(np.array(x_channels).T)

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        print(dataset_name, 'max', max_length, 'min', min_length)

        if min_length != max_length:
            x_train = transform_to_same_length(x_train, max_length)
            x_test = transform_to_same_length(x_test, max_length)

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')


if __name__ == '__main__':
    mts_root_dir = 'I:/Downloads/Multivariate2018_ts/'
    mts_out_dir = './archives/Multivariate2018_ts/'
    transform_sktime_to_npy_format(mts_root_dir, mts_out_dir)
