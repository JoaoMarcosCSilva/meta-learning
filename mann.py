import tensorflow as tf
from tensorflow.keras.layers import *

def MANN(num_classes = 5, num_samples_per_class = 2, layers = [LSTM(128, return_sequences = True), LSTM(128, return_sequences = True)]):
    images = Input(shape = (num_samples_per_class, num_classes, 28, 28, 1))
    labels = Input(shape = (num_samples_per_class, num_classes, num_classes))

    im = tf.reshape(images, (-1, num_samples_per_class * num_classes, 28*28))
    
    lbl = tf.zeros_like(labels[:,-1:])
    

    lbl = tf.concat([labels[:,0:-1], lbl], 1)
    lbl = tf.reshape(lbl, (-1, num_samples_per_class * num_classes, num_classes))

    x = tf.concat([im, lbl], -1)
    
    for l in layers:
        x = l(x)
    
    x = Dense(num_classes)(x)
    x = Softmax()(x)

    x = tf.reshape(x, (-1, num_samples_per_class, num_classes, num_classes))

    outputs = x

    def loss(y_true, y_pred):
        preds = y_pred[:, -1]
        labels = y_true[:, -1]
        return tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(from_logits = False)(labels, preds))

    def accuracy (y_true, y_pred):
        labels = y_true[:, -1]
        preds = y_pred[:, -1]
        return tf.keras.metrics.categorical_accuracy(labels, preds)

    model = tf.keras.Model([images, labels], outputs)
    model.compile(loss = loss, optimizer = 'Adam', metrics = [accuracy])

    return model
