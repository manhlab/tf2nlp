from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


def build_model(transformer, max_len):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    sequence_output = transformer(input_ids)[0]
    cls_token = sequence_output[:, 0, :]
    cls_token = tf.reduce_mean(
        tf.stack([Dropout(i)(cls_token) for i in np.arange(0.1, 0.5, 0.1)], axis=0),
        axis=0,
    )
    cls_token = Dense(32, activation="relu")(cls_token)
    out = Dense(3, activation="softmax")(cls_token)

    model = Model(inputs=input_ids, outputs=out)
    model.compile(
        Adam(lr=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
