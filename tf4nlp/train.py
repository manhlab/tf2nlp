import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import warnings
import os
import config
from model import build_model
from utils import quick_encode, create_dataset, plot_history
warnings.filterwarnings('ignore')

print(os.listdir(config.PATH))
df_train = pd.read_csv(os.path.join(config.PATH, "train_translate_en.csv"))
df_test = pd.read_csv(os.path.join(config.PATH,"test_translate_en.csv"))

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# Our batch size will depend on number of replic
BATCH_SIZE= config.BATCH_SIZE * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE
tokenizer = AutoTokenizer.from_pretrained(config.MODEL)

x_train = quick_encode(df_train, tokenizer)
x_test = quick_encode(df_test, tokenizer)
y_train = df_train.label.values

test_dataset = (tf.data.Dataset.from_tensor_slices((x_test))).batch(BATCH_SIZE)


pred_test = np.zeros((df_test.shape[0],3))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_score = []
history = []
pred = []
for fold, (train_ind, valid_ind) in enumerate(skf.split(x_train, y_train)):
        print("fold",fold+1)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        train_data = create_dataset(x_train[train_ind],y_train[train_ind],val=False)
        valid_data = create_dataset(x_train[valid_ind],y_train[valid_ind],val=True)
    
        Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"{config.MODEL}-{fold}.h5", monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='min')
        
        with strategy.scope():
            transformer_layer = TFAutoModel.from_pretrained(config.MODEL)
            model = build_model(transformer_layer, max_len=config.MAX_LEN)
        n_steps = len(train_ind)//BATCH_SIZE
        print("training model {} ".format(fold+1))

        train_history = model.fit(
        train_data,
        steps_per_epoch=n_steps,
        validation_data=valid_data,
        epochs=config.EPOCHS,callbacks=[Checkpoint],verbose=1)
        
        print("Loading model...")
        model.load_weights(f"{config.MODEL}-{fold}.h5")
        print("fold {} validation accuracy {}".format(fold+1,np.mean(train_history.history['val_accuracy'])))
        print("fold {} validation loss {}".format(fold+1,np.mean(train_history.history['val_loss'])))
        
        history.append(train_history)

        val_score.append(np.mean(train_history.history['val_accuracy']))
        
        print('predict on test....')
        preds=model.predict(test_dataset,verbose=1)
        pred.append(preds)


plot_history(history, config.EPOCHS)