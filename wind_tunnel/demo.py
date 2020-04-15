# Tested in TensorFlow==2.0.0

import functools
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.metrics import MeanMetricWrapper


def load_data(file):
    df = pd.read_csv(file, sep='\t')
    mach = np.array(df.columns)[1:].astype(np.float)
    alpha = np.array(df['Alpha\\Mach']).astype(np.float)
    values = np.array(df)[:, 1:].astype(np.float)

    con_1, con_2, con_3 = tuple(os.path.basename(file).strip()[:-4].split('_'))
    con_1 = np.array([[float(con_1)]] * np.size(values))
    con_2 = np.array([[float(con_2)]] * np.size(values))
    con_3 = np.array([[float(con_3)]] * np.size(values))

    mach, alpha = tuple(np.meshgrid(mach, alpha))
    mach = np.reshape(mach, (-1, 1))
    alpha = np.reshape(alpha, (-1, 1))
    values = np.reshape(values, (-1, 1))
    np_dataset = np.concatenate([con_1, con_2, con_3, alpha, mach, values], axis=-1)
    return np_dataset


def make_np_dataset(pattern):
    files = tf.io.gfile.glob(pattern)
    np_datasets = list(map(load_data, files))
    np_dataset = np.concatenate(np_datasets, axis=0)

    x = np_dataset[:, :-1]
    y = np_dataset[:, -1:]
    return x, y


def mean_scaled_relative_error(y_true, y_pred, epsilon=1.):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.abs(y_pred - y_true) / (tf.abs(y_true) + epsilon), axis=-1)


class MeanScaledRelativeError(MeanMetricWrapper):
    def __init__(self, epsilon, name='mean_scaled_relative_error', dtype=None):
        super(MeanScaledRelativeError, self).__init__(
            functools.partial(mean_scaled_relative_error, epsilon=epsilon), name, dtype=dtype)


def build_demo_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((5,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(1)])
    model.compile(
        tf.keras.optimizers.Adam(0.0015),  # tf.keras.optimizers.SGD(0.001, 0.99),
        loss='mse',
        metrics=[MeanScaledRelativeError(0.01), 'mse', 'mae'])
    return model


def train(train_files, test_files,
          saved_model_dir=None,
          saved_tensorboard_dir=None,
          saved_history_file=None):
    train_x, train_y = make_np_dataset(train_files)
    test_x, test_y = make_np_dataset(test_files)
    model = build_demo_model()
    callbacks = []
    if saved_model_dir is not None:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                saved_model_dir, monitor='val_mae', save_best_only=True))
    if saved_tensorboard_dir is not None:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=saved_tensorboard_dir,
                histogram_freq=0, write_graph=True, write_images=True,
                update_freq='epoch', profile_batch=2))
    history = model.fit(
        train_x, train_y,
        batch_size=1028,
        epochs=1000,
        validation_data=(test_x, test_y),
        shuffle=True,
        callbacks=callbacks)
    if saved_history_file is not None:
        pd.DataFrame(history.history).to_csv(saved_history_file, index=False)
    return model, history


def predict(model_or_file, test_files, saved_prediction_file=None):
    if type(model_or_file) is str:
        model = build_demo_model()
        model.load_weights(model_or_file)
    elif isinstance(model_or_file, tf.keras.Model):
        model = model_or_file
    else:
        raise ValueError('`model_or_dir` must be a model or a checkpoint file.')
    test_x, test_y = make_np_dataset(test_files)
    y_pred = model.predict(test_x)
    if saved_prediction_file is None:
        return y_pred
    pd.DataFrame(
        np.concatenate([test_x, test_y, y_pred], axis=-1)
    ).to_csv(
        saved_prediction_file, header=False, index=False)


def main():
    test_files = ['data/numerical/5750_0.30_0.40.txt']
    train_files = list(filter(
        lambda f: f not in tf.io.gfile.glob(test_files), tf.io.gfile.glob('data/numerical/*.txt')))
    train(
        train_files, test_files,
        saved_model_dir='saved/ckpt',
        saved_tensorboard_dir='saved\\tensorboard',
        saved_history_file='saved/history.csv')
    predict('saved/ckpt/variables/variables', test_files, 'saved/prediction.csv')
    # predict('saved/best.ckpt', test_files, 'saved/prediction.csv')  # for TensorFlow 1.x


if __name__ == '__main__':
    main()
