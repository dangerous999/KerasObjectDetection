import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras.metrics as metrics

from keras.models import Model, load_model, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.optimizers import SGD, Adamax, Adam, RMSprop
from layer_visualization import visualize_layer

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#converter = tf.lite.TFLiteConverter.from_keras_model_file('mobnetv2_fine_tune_categorical_ADAM_no_metrics.h5')
#tfmodel = converter.convert()
#
#open("mobnetv2_fine_tune_categorical_ADAM_metrics.tflite" , "wb").write(tfmodel)
#
#file = open( 'mobnetv2_fine_tune_categorical_ADAM_metrics.tflite' , 'wb' )
#file.write(model)


IMAGE_SIZE = 224
NUM_CHANNELS = 3
BATCH_SIZE = 16

EPOCHS = 1

VALIDATION_STEPS = 64

LEARNING_RATE = 1e-3

# ---------------------------- DATA GENERATORS ----------------------------

TRAIN_DIR = 'datasets/kaggle_original_glasses_table/train'
VALIDATION_DIR = 'datasets/kaggle_original_glasses_table/test'
TEST_DIR = 'datasets/kaggle_original_glasses_table/test'

CLASS_MODE = "binary"
MODEL_METRICS = [metrics.binary_accuracy,
                 metrics.Precision(),
                 metrics.Recall(),
                 metrics.MeanAbsoluteError(),
                 metrics.MeanSquaredError(), 
                 metrics.TruePositives(),
                 metrics.TrueNegatives(), 
                 metrics.FalsePositives(),
                 metrics.FalseNegatives()
                 ]

def get_train_data():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255.,
        fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE)
    return train_generator


def get_validation_data():
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE)
    return validation_generator


def get_test_data():
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=16,
        class_mode=CLASS_MODE,
        shuffle=True)
    return test_generator


def get_mobile_net_model(include_top=False):
    base_model =  MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                              include_top=False,
                              weights="imagenet")
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
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds


def plot_preds(img, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        preds: list of predicted labels and their probabilities
    """
    labels = ("glass", "table")
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.figure(figsize=(8,8))
    plt.subplot(gs[0])
    plt.imshow(np.asarray(img))
    plt.subplot(gs[1])
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


def build_model():
    model = keras.models.Sequential()
    model.add(get_mobile_net_model())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # 1 when binary, 2 when categorical

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=MODEL_METRICS)
    return model


def plot_training_history(history):
    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()

    # plot metrics
    # plt.plot(history.history['acc'])
    # plt.show()


if __name__=='__main__':

    train_generator = get_train_data()
    validation_generator = get_validation_data()
    test_generator = get_test_data()

    model = build_model()
    history = model.fit_generator(train_generator,
                                  epochs=EPOCHS,
                                  steps_per_epoch=train_generator.n / BATCH_SIZE,
                                  validation_data=validation_generator,
                                  validation_steps=VALIDATION_STEPS)
    model.save('mobnetv2_fine_tune_categorical_ADAM_no_metrics.h5')
    plot_training_history(history)


    model = load_model('mobnetv2_fine_tune_categorical_ADAM_no_metrics.h5', compile=False)
    # model = load_model('mobnetv2_fine_tune_categorical_ADAM.h5', custom_objects={'precision': keras.metrics.Precision})

    # img = image.load_img('datasets/kaggle_original_glasses_table/test/glass/glass_3000.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # img = image.load_img('datasets/kaggle_original_glasses_table/test/table/table_3000.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # preds = predict(model, img)
    # print(preds)
    # plot_preds(np.asarray(img), preds)

    # Confusion Matrix and Classification Report
    # img = image.load_img("datasets\kaggle_original_glasses_table/test/glass/glass_3000.jpg", target_size=(IMAGE_SIZE, IMAGE_SIZE))

    # print(predict(model, img))
    # plot_preds(img, predict(model, img))

    predictions = model.predict(test_generator, verbose=1)
    for prediction in predictions:
        print(prediction)

    # y_pred = np.argmax(Y_pred, axis=1)
    # print(y_pred)
    # print('Confusion Matrix')
    # print(test_generator.classes, y_pred)
    # print(confusion_matrix(validation_generator.classes, y_pred))
    # print('Classification Report')
    # target_names = ['table', 'glass']
    # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))