import tensorflow as tf
import numpy as np


def predict_with_model(model, img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, [60,60]) # (60,60,3)
    img = tf.expand_dims(img, axis=0) # (1,60,60,3)

    predictions = model.predict(img) # Returns list of probabilities of img belonging to each class
    predictions = np.argmax(predictions) # Index of max value

    return predictions

if __name__=='__main__':
    img_path = '/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/Test/2/00409.png'
    img_path = '/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/Test/0/00807.png'
    model = tf.keras.models.load_model('/Users/masonscott/git-repos/Python-Projects/Tensorflow-Tutorial/data/Models.keras')
    prediction = predict_with_model(model, img_path)

    signs = ['20 KMH', '30 KMH', '50 KMH', '60 KMH', '70 KMH', '80 KMH', '', '', '', '']

    print(f'prediction = {signs[prediction]}')