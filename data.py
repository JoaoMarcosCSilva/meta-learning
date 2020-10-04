import random
import os
import imageio
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

def get_task_filenames(folders, n_classes, n_samples_per_class):
    f = random.sample(folders, n_classes)
    
    l = []

    for class_id, class_folder in enumerate(f):
        files = random.sample(os.listdir(class_folder), n_samples_per_class)
        files = [(os.path.join(class_folder, i), class_id) for i in files]
        l += files

    return l

def load_image(path):
    return (1 - (imageio.imread(path) / 255)).astype(np.float32)

def get_folders(num_train = 1100, num_val = 100):
    alphabet_folders = [os.path.join('./omniglot_resized', i) for i in os.listdir('./omniglot_resized')]
    folders = [os.path.join(i, j) for i in alphabet_folders for j in os.listdir(i)]
    
    train = folders[:num_train]
    val = folders[num_train:num_train + num_val]
    test = folders[num_train + num_val:]

    return train, val, test

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folders, n_classes = 5, n_samples_per_class = 2, n_steps = 1000, batch_size = 16):
       super(DataGenerator, self).__init__()
        
       self.folders = folders
       self.n_classes = n_classes
       self.n_samples_per_class = n_samples_per_class
       self.n_steps = n_steps
       self.batch_size = batch_size

    def __len__(self):
        return self.n_steps

    # Returns (x_support, y_support), (x_query, y_query)
    def __getitem__(self, it):

        # Contains task files for all batches
        files = [get_task_filenames(self.folders, self.n_classes, self.n_samples_per_class) for i in range(self.batch_size)]
        
        all_images = []
        all_labels = []

        # Iterate for each batch
        for batch_files in files:
            # Loads each image
            images = np.stack([load_image(p[0]) for p in batch_files])
            images = np.reshape(images, (self.n_samples_per_class, self.n_classes, 28, 28))
            
            # Converts the labels to categorical
            labels = [p[1] for p in batch_files]
            labels = tf.keras.utils.to_categorical(labels, self.n_classes)
            labels = np.reshape(labels, (self.n_samples_per_class, self.n_classes, self.n_classes), 'F')
            
            last_images = images[-1]
            last_labels = labels[-1]

            last_images, last_labels = shuffle(last_images, last_labels)
            

            images[-1] = last_images
            labels[-1] = last_labels

            all_images.append(images)
            all_labels.append(labels)

        all_images = np.stack(all_images)
        all_labels = np.stack(all_labels)

        all_images = np.expand_dims(all_images, -1)

        return ((all_images, all_labels), all_labels)
