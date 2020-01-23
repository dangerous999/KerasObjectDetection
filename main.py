import os
import numpy as np
import matplotlib.pyplot as plt
import keras.metrics as metrics

from keras.models import load_model, Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.vgg19 import VGG19

IMAGE_SIZE = 224
NUM_CHANNELS = 3

EPOCHS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
CLASS_MODE = "binary"
MODEL_METRICS = [metrics.binary_accuracy,
                 metrics.Precision(),
                 metrics.Recall(),
                 metrics.MeanAbsoluteError(),
                 metrics.MeanSquaredError(),
                 metrics.TruePositives(),
                 metrics.TrueNegatives(),
                 metrics.FalsePositives(),
                 metrics.FalseNegatives()]

# ---------------------------- DATA GENERATORS ----------------------------

TRAIN_DIR = 'datasets/tennis_ball/emptyballs/train'
VALIDATION_DIR = 'datasets/tennis_ball/emptyballs/test'
TEST_DIR = 'datasets/tennis_ball/emptyballs/test'


def get_train_data(preprocess_input=None):
    if preprocess_input is not None:
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255.,
            fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            vertical_flip=True,
            horizontal_flip=True,
            rescale=1./255.,
            fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE)
    return train_generator


def get_validation_data(preprocess_input=None):
    if preprocess_input is not None:
        validation_datagen = ImageDataGenerator(
            rotation_range=40,
            vertical_flip=True,
            rescale=1./255.,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        validation_datagen = ImageDataGenerator(
            rotation_range=40,
            vertical_flip=True,
            rescale=1./255.,
            horizontal_flip=True,
            fill_mode='nearest')

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE)
    return validation_generator


def get_test_data():
    test_datagen = ImageDataGenerator(rescale=1./255.)  # )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=16,
        class_mode=CLASS_MODE,
        shuffle=False)
    return test_generator


def get_mobile_net_model(include_top=False):
    base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                             include_top=False,
                             weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False

    return base_model


def get_vgg19_net_model(include_top=False):
    base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                       include_top=False,
                       weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    return base_model


def predict(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities
    """
    preds = model.predict(img)
    return preds


def build_model():
    model = Sequential()
    model.add(get_mobile_net_model())  # model.add(get_vgg19_net_model())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # 1 when binary, 2 when categorical
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=MODEL_METRICS)
    return model


def plot_training_history(history):
    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(EPOCHS))
    plt.legend(['loss'], loc='upper left')

    plt.show()

    plt.plot(history.history['binary_accuracy'], label="accuracy")
    plt.plot(history.history['precision_1'], label="precision")
    plt.xticks(range(EPOCHS))
    plt.legend()
    plt.show()

    plt.plot(history.history['val_binary_accuracy'], label="val_accuracy")
    plt.plot(history.history['val_precision_1'], label="val_precision")
    plt.xticks(range(EPOCHS))
    plt.legend()
    plt.show()


if __name__ == '__main__':

    train_generator = get_train_data(preprocess_input)
    validation_generator = get_validation_data(preprocess_input)

    model = build_model()

    history = model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_generator.n / BATCH_SIZE,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n / BATCH_SIZE)

    model.save(model.name + "_" + str(LEARNING_RATE) + ".h5")

    plot_training_history(history)

    model = load_model('mobilenet_rescale.h5', compile=False)

    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_datagen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=16,
        class_mode=CLASS_MODE,
        shuffle=True)

    for batch in test_datagen:
        results = predict(model, batch[0])
        fig, axs = plt.subplots(4, 4, facecolor='w', edgecolor='k')
        fig.subplots_adjust(wspace=.2)
        axs = axs.ravel()
        for i in range(16):
            prob = round(1 - results[i][0], 2)
            if prob > 0.5:
                axs[i].set_title("Ball " + str(prob))
            else:
                axs[i].set_title("None " + str(prob))

            axs[i].imshow(batch[0][i])
            axs[i].tick_params(which='both', bottom=False, top=False,
                               left=False, labelbottom=False, labelleft=False)
        font = {'family': 'normal',
                'size': 25}
        plt.rc('font', **font)
        plt.show()


# converter = tf.lite.TFLiteConverter.from_keras_model_file(‘keras_model.h5')
# tfmodel = converter.convert()
# file = open("model.tflite" , "wb").write(tfmodel)
# file.write(model)
