import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from my_utils import split_data, order_test_set, create_generators
from deep_learning_models import street_signs_model

if __name__=='__main__':
    path_to_train = '/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/training_data/train'
    path_to_val = '/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/training_data/val'
    path_to_test = '/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/Test'
    batch_size = 64
    epochs=15
    lr = 0.001 # default learning rate

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    num_classes = train_generator.num_classes

    TRAIN = False
    TEST = True


    if TRAIN:
        # Saves the best model from the validation accuracy during training
        checkpoint_path = '/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/Models.keras'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_saver = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            # max for val_accuracy, min for val_loss
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

        model = street_signs_model(num_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[checkpoint_saver, early_stop]
        )

    if TEST:
        model = tf.keras.models.load_model('/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/Models.keras')
        model.summary()

        print('Evaluating Validation Set:')
        model.evaluate(val_generator)

        print('Evaluating Test Set:')
        model.evaluate(test_generator) 
