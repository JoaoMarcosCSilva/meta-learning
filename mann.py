import tensorflow as tf
from tensorflow.keras.layers import *

def MANN(num_classes = 5, num_samples_per_class = 2):
    images = Input(shape = (num_samples_per_class, num_classes, 28, 28, 1))
    labels = Input(shape = (num_samples_per_class, num_classes, num_classes))

    im = tf.reshape(images, (-1, num_samples_per_class * num_classes, 28*28))
    
    lbl = tf.zeros_like(labels[:,-1:])
    

    lbl = tf.concat([labels[:,0:-1], lbl], 1)
    lbl = tf.reshape(lbl, (-1, num_samples_per_class * num_classes, num_classes))

    x = tf.concat([im, lbl], -1)

#    x = Bidirectional(LSTM(512, return_sequences = True))(x)
#    x = Bidirectional(LSTM(512, return_sequences = True))(x)
    x = LSTM(128, return_sequences = True)(x)
    x = LSTM(128, return_sequences = True)(x)
    x = Dense(512, activation = 'relu')(x)
    
    x = Dense(num_classes)(x)
    x = Softmax()(x)

    x = tf.reshape(x, (-1, num_samples_per_class, num_classes, num_classes))

    outputs = x

    def loss(y_true, y_pred):
        preds = y_pred[:, -1]
        labels = y_true[:, -1]
        return tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(from_logits = False)(labels, preds))

    model = tf.keras.Model([images, labels], outputs)
    model.compile(loss = loss, optimizer = 'Adam')

    return model
