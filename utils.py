import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
import os

def quick_encode(df,tokenizer, maxlen=100):
    
    values = df[['premise','hypothesis']].values.tolist()
    tokens=tokenizer.batch_encode_plus(values,max_length=maxlen,
                                       truncation=True,
                                       pad_to_max_length=True)
    
    return np.array(tokens['input_ids'])

def create_dataset(X ,y ,val, batch_size, AUTO):
    dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(X))
    if not val:
        dataset = dataset.repeat().batch(batch_size).prefetch(AUTO)
    else:
        dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset

def plot_history(history, EPOCHS):
    plt.figure(figsize=(15,10))

    for i,hist in enumerate(history):

        plt.subplot(3,2,i+1)
        plt.plot(np.arange(EPOCHS),hist.history['accuracy'],label='train accu')
        plt.plot(np.arange(EPOCHS),hist.history['val_accuracy'],label='validation acc')
        plt.gca().title.set_text(f'Fold {i+1} accuracy curve')
        plt.legend()