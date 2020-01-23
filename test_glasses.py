import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras.metrics as metrics

from keras.models import Model, load_model, model_from_json, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.vgg19 import VGG19
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
# file.write(model)


IMAGE_SIZE = 224
NUM_CHANNELS = 3
BATCH_SIZE = 16

EPOCHS = 20

VALIDATION_STEPS = 64

LEARNING_RATE = 1e-3


# ---------------------------- DATA GENERATORS ----------------------------

# TRAIN_DIR = 'datasets/kaggle_original_glasses_table/train'
# VALIDATION_DIR = 'datasets/kaggle_original_glasses_table/test'
# TEST_DIR = 'datasets/kaggle_original_glasses_table/test'

TRAIN_DIR = 'datasets/tennis_ball/emptyballs/train'
VALIDATION_DIR = 'datasets/tennis_ball/emptyballs/test'
TEST_DIR = 'datasets/tennis_ball/emptyballs/test'

CLASS_MODE = "binary"
MODEL_METRICS = [metrics.binary_accuracy,
                 metrics.Precision(),
                 # metrics.Recall(),
                 metrics.MeanAbsoluteError(),
                 metrics.MeanSquaredError(),
                 metrics.TruePositives(),
                 metrics.TrueNegatives(),
                 metrics.FalsePositives(),
                 metrics.FalseNegatives()]


def get_train_data(preprocess_input=None):
    if preprocess_input is not None:
        train_datagen = ImageDataGenerator(
            # preprocessing_function=preprocess_input,
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
            # preprocessing_function=preprocess_input,
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
    # x = img.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    preds = model.predict(img)
    return preds


def plot_preds(img, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        preds: list of predicted labels and their probabilities
    """
    labels = ("glass", "table")
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.figure(figsize=(8, 8))
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
    model = Sequential()
    model.add(get_mobile_net_model())
    # model.add(get_vgg19_net_model())

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

    print(history.history)

    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(EPOCHS))
    plt.legend(['loss'], loc='upper left')

    savefig('foo.png', bbox_inches='tight')
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

    # plot metrics
    # plt.plot(history.history['acc'])
    # plt.show()
    # TODO history.history dict contains all values through training (use it for graphs)


if __name__ == '__main__':
    test_generator = get_test_data()

    '''

    # ---------------- LOSS ----------------
    adam_mobile_net_train_1e1_loss = [3119.986771658878, 19.000732200331125, 0.7195164041994264, 0.7056519022688976, 0.7698043445046009, 1.5469784130447546, 0.730100053411208, 0.7039889699778319, 1.2670202762938647, 1.9530621038462384, 0.7175732379550632, 0.7312983386716461, 0.7468077181900692, 0.7487186357867819, 0.7160481738146136, 0.7612325716424249, 0.7076358440290245, 0.8001508864858171, 0.7929009285326571, 0.7301879168136152]
    adam_mobile_net_train_1e2_loss = [23.743890718295535, 0.5812077921332015, 0.605006074177538, 0.45470448354324766, 0.429923820466636, 0.4209484611869872, 0.47483804593833656, 0.3922329932125342, 0.33204406297076516, 0.3311053589529574, 0.36297992422253367, 0.3629453343363028, 0.3028812238613861, 0.3615527623044474, 0.3641238654007454, 0.3779540106491214, 0.3382425760513088, 0.35290582244480223, 0.30470780005286996, 0.2785483057546188]
    adam_mobile_net_train_1e3_loss = [
                                    1.6807182872280895,
                                    0.6413412393950025,
                                    0.4799588818741308,
                                    0.5108301319534985,
                                    0.46624111415838704,
                                    0.41924521016502436,
                                    0.4473961744534433,
                                    0.3723219953576965,
                                    0.31892448239128035,
                                    0.3397881243454874,
                                    0.32235529519083433,
                                    0.3342955142886515,
                                    0.3151210635574591,
                                    0.2955722340257026,
                                    0.32674616528165734,
                                    0.3382420983360141,
                                    0.30547032655552847,
                                    0.29636998045487944,
                                    0.2843875879034816,
                                    0.3563627276936249]

    adam_mobile_net_val_1e1_loss = [0.687555193901062, 0.663806676864624, 0.6516425609588623, 0.7109421491622925, 0.7111935019493103, 0.9092508554458618, 0.6948649883270264, 0.7354710102081299, 0.7642793655395508, 0.6962906122207642, 0.6776100993156433, 0.6979880332946777, 0.6860901713371277, 0.6871267557144165, 0.9511807560920715, 0.6691840887069702, 0.6685234904289246, 0.9056893587112427, 0.6622401475906372, 0.7283195853233337]
    adam_mobile_net_val_1e2_loss = [0.6506805419921875, 0.282303124666214, 0.7128006219863892, 0.7146787643432617, 0.6004487872123718, 0.5014482140541077, 0.5679303407669067, 0.6972172260284424, 0.15545392036437988, 0.8171029090881348, 0.9098899364471436, 0.6843925714492798, 0.8527529239654541, 0.516514241695404, 1.0465368032455444, 0.44556164741516113, 1.1908472776412964, 0.4393766224384308, 0.5894021987915039, 0.8084275126457214]
    adam_mobile_net_val_1e3_loss = [
        1.1161465644836426,
        0.16466103494167328,
        0.4964078962802887,
        0.6524842381477356,
        0.49869686365127563,
        0.3492799401283264,
        0.28819558024406433,
        0.7594720125198364,
        0.5468566417694092,
        0.4488478899002075,
        1.5087547302246094,
        1.2276294231414795,
        0.3356684446334839,
        0.6425526142120361,
        0.33934685587882996,
        0.31501346826553345,
        0.2009844332933426,
        0.4957295358181,
        0.4722062349319458,
        0.4518774747848511]


    # ---------------- PRECISION ----------------
    adam_mobile_net_train_1e1_precision = [0.47699758, 0.519802, 0.48535565, 0.51983297, 0.496788, 0.50627613, 0.4851936, 0.509434, 0.4986945, 0.49203187, 0.51458335, 0.5329087, 0.50827426, 0.48387095, 0.51566267, 0.4861996, 0.5037594, 0.51536644, 0.51111114, 0.5]
    adam_mobile_net_train_1e2_precision = [0.6216216, 0.68907565, 0.7702703, 0.7905405, 0.8041958, 0.80516434, 0.8038793, 0.8236659, 0.8355856, 0.8668224, 0.82366073, 0.8053097, 0.8319328, 0.765873, 0.87593055, 0.8248848, 0.85211265, 0.8463303, 0.8551402, 0.87198067]
    adam_mobile_net_train_1e3_precision = [
                                            0.66423357,
                                            0.7679426,
                                            0.79196215,
                                            0.7692308,
                                            0.8108747,
                                            0.8175355,
                                            0.8173302,
                                            0.82435596,
                                            0.8679245,
                                            0.86374694,
                                            0.85876995,
                                            0.8666667,
                                            0.8503401,
                                            0.8907767,
                                            0.8627907,
                                            0.8627907,
                                            0.8666667,
                                            0.87142855,
                                            0.88106793,
                                            0.85287356]

    adam_mobile_net_val_1e1_precision = [0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
    adam_mobile_net_val_1e2_precision = [0.774193525314331, 0.7749999761581421, 0.738095223903656, 0.8484848737716675, 0.8947368264198303, 0.8387096524238586, 0.75, 0.8666666746139526, 0.8222222328186035, 0.8518518805503845, 0.8500000238418579, 0.7674418687820435, 0.8787878751754761, 0.807692289352417, 0.7749999761581421, 0.7647058963775635, 0.95652174949646, 0.875, 0.7878788113594055, 0.8695651888847351]
    adam_mobile_net_val_1e3_precision = [
                    0.8399999737739563,
                    0.8292682766914368,
                    0.7400000095367432,
                    0.8055555820465088,
                    0.8709677457809448,
                    0.7954545617103577,
                    0.8484848737716675,
                    0.8666666746139526,
                    0.8846153616905212,
                    0.9642857313156128,
                    0.8333333134651184,
                    0.6724137663841248,
                    0.8285714387893677,
                    0.7346938848495483,
                    0.837837815284729,
                    0.875,
                    0.7872340679168701,
                    0.8292682766914368,
                    0.8500000238418579,
                    0.7777777910232544]

    # ---------------- ACCURACY ----------------
    adam_mobile_net_val_1e1_accuracy = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    adam_mobile_net_val_1e2_accuracy = [0.7124999761581421, 0.7749999761581421, 0.75, 0.7875000238418579, 0.6875, 0.762499988079071, 0.675000011920929, 0.637499988079071, 0.862500011920929, 0.737500011920929, 0.675000011920929, 0.7875000238418579, 0.8125, 0.699999988079071, 0.7749999761581421, 0.7250000238418579, 0.762499988079071, 0.7250000238418579, 0.737500011920929, 0.7124999761581421]
    adam_mobile_net_val_1e3_accuracy = [
                            0.7124999761581421,
                            0.8374999761581421,
                            0.800000011920929,
                            0.7749999761581421,
                            0.7875000238418579,
                            0.824999988079071,
                            0.7875000238418579,
                            0.7749999761581421,
                            0.75,
                            0.824999988079071,
                            0.75,
                            0.75,
                            0.7875000238418579,
                            0.7875000238418579,
                            0.8125,
                            0.875,
                            0.8374999761581421,
                            0.8374999761581421,
                            0.8500000238418579,
                            0.8125]

    adam_mobile_net_train_1e1_accuracy = [0.47023088, 0.5127582, 0.4763062, 0.5164034, 0.48967195, 0.50060755, 0.47752127, 0.5030377, 0.49210206, 0.4835966, 0.51032805, 0.5309842, 0.5018226, 0.47509113, 0.509113, 0.47752127, 0.49696234, 0.509113, 0.5030377, 0.49331713]
    adam_mobile_net_train_1e2_accuracy = [0.6136088, 0.71202916, 0.78493315, 0.80680436, 0.8104496, 0.8092345, 0.835966, 0.83232075, 0.85540706, 0.8748481, 0.8456865, 0.82867557, 0.87727827, 0.81895506, 0.8614824, 0.835966, 0.8578372, 0.86026734, 0.8626974, 0.8675577]
    adam_mobile_net_train_1e3_accuracy = [
                                    0.65735114,
                                    0.7654921,
                                    0.7934386,
                                    0.77399755,
                                    0.8128797,
                                    0.81895506,
                                    0.82260025,
                                    0.82989067,
                                    0.872418,
                                    0.8566221,
                                    0.87606317,
                                    0.8675577,
                                    0.8687728,
                                    0.88456863,
                                    0.872418,
                                    0.872418,
                                    0.8675577,
                                    0.872418,
                                    0.8748481,
                                    0.86634266]
    '''
    '''
    # ----------------- 256x256x256x256x1 -----------------

    # ---------------------- ADAM -------------------------

    # mobnet 0.001
    adam_mobile_net_val_1e3_loss = [0.539158046245575, 0.29042181372642517, 0.5680433511734009, 0.32944828271865845, 0.44658833742141724, 0.8245891332626343, 0.2864941656589508, 0.23030422627925873, 0.37851089239120483, 0.4679161012172699, 0.3842942416667938, 0.4943596124649048, 0.36922746896743774, 0.03999517858028412, 0.3064793646335602, 0.5163618326187134, 0.32750654220581055, 0.3997248709201813, 0.5886750817298889, 0.2347998321056366]
    adam_mobile_net_val_1e3_accuracy = [0.6875, 0.8125, 0.7250000238418579, 0.7749999761581421, 0.800000011920929, 0.7749999761581421, 0.7749999761581421, 0.8125, 0.8125, 0.8500000238418579, 0.8374999761581421, 0.8374999761581421, 0.824999988079071, 0.875, 0.8500000238418579, 0.8374999761581421, 0.824999988079071, 0.8999999761581421, 0.8374999761581421, 0.7875000238418579]

    adam_mobile_net_train_1e3_loss = [1.1865988683005295, 0.4891764003588826, 0.40280905083491764, 0.41701387569652293, 0.4318997683965421, 0.36929602332567096, 0.2951831915398849, 0.3480163484920418, 0.2826725450929431, 0.29137926359512684, 0.26112768067587916, 0.250539697788176, 0.2763026765981103, 0.2467930257537675, 0.22679570979944555, 0.23462172864708836, 0.2546584396866373, 0.2006675305155039, 0.17952255932286038, 0.19669601776810275]
    adam_mobile_net_train_1e3_accuracy = [0.6257594, 0.7946537, 0.8238153, 0.8262454, 0.835966, 0.8396112, 0.87727827, 0.84933174, 0.9003645, 0.8797084, 0.8979344, 0.9003645, 0.8869988, 0.89671934, 0.90765494, 0.9052248, 0.8942892, 0.9161604, 0.9222357, 0.927096]

    # mobnet 0.01
    adam_mobile_net_val_1e2_loss = [0.31583720445632935, 0.4379592537879944, 0.2861841320991516, 0.6482062339782715, 1.09665048122406, 0.2465689480304718, 0.27970457077026367, 0.6618553400039673, 0.23573976755142212, 0.6115996241569519, 0.6943787336349487, 0.3447035253047943, 0.6469142436981201, 0.6138250827789307, 0.5491403937339783, 0.24410831928253174, 0.5131736993789673, 0.13457657396793365, 0.40279364585876465, 0.5146185755729675]
    adam_mobile_net_val_1e2_accuracy = [0.7250000238418579, 0.762499988079071, 0.800000011920929, 0.8125, 0.8125, 0.8374999761581421, 0.8374999761581421, 0.800000011920929, 0.8374999761581421, 0.824999988079071, 0.42500001192092896, 0.800000011920929, 0.5, 0.675000011920929, 0.762499988079071, 0.737500011920929, 0.625, 0.824999988079071, 0.762499988079071, 0.8125]

    adam_mobile_net_train_1e2_loss = [8.372402488026555, 0.5763854419547761, 0.46785719756077737, 0.5436569032952968,  0.47183096821658665, 0.5028504202751168, 0.3874954765216693, 0.35000810577542063, 0.39367972412129654, 0.3796368843874694, 0.4836397358959819, 0.5411296132703018, 0.6417738776540843, 0.626108890894986, 0.5098284842177121,    0.5078543944607963, 0.3612873314507532, 0.4478067779055777, 0.5099381230146888, 0.4001168790448769]
    adam_mobile_net_train_1e2_accuracy = [0.55771565, 0.72782505, 0.7934386, 0.77521265, 0.8055893, 0.7934386, 0.835966, 0.8505468, 0.8347509,  0.8517618, 0.76063186, 0.70595384, 0.7144593, 0.6354799, 0.80680436, 0.80194414, 0.8578372, 0.8165249, 0.81773996, 0.8639125]

    # vgg19 0.001
    adam_vgg19_val_1e3_loss = [0.41729259490966797, 0.35387304425239563, 0.25756266713142395, 0.45707303285598755, 0.3911333680152893, 0.37945425510406494, 0.08279848098754883, 0.27605772018432617, 0.2860063910484314, 0.43099841475486755, 0.20386601984500885, 0.21050389111042023, 0.139244943857193, 0.14000102877616882, 0.08394599705934525, 0.3322674632072449, 0.5600109100341797, 0.293287068605423, 0.41083449125289917, 0.08063505589962006]
    adam_vgg19_val_1e3_accuracy = [0.6625000238418579, 0.8500000238418579, 0.824999988079071, 0.862500011920929, 0.862500011920929, 0.8500000238418579, 0.8999999761581421, 0.875, 0.800000011920929, 0.887499988079071, 0.9125000238418579, 0.8999999761581421, 0.9375, 0.9125000238418579, 0.9375, 0.75, 0.8500000238418579, 0.875, 0.875, 0.8999999761581421]

    adam_vgg19_train_1e3_loss = [0.6911792586379903, 0.470001253293757, 0.3943400007841219, 0.30098120464444306, 0.3535186594232175, 0.2882096107646298, 0.22599557262334977, 0.2664985037267317, 0.24001543698232636, 0.20442117422927772, 0.20642444316664413, 0.19300112251830565, 0.17606522696475213, 0.17491021469000043, 0.17720707281428147, 0.2589458228334477, 0.17318077836262644, 0.15398202255766563, 0.17457652705657054, 0.1482545275462934]
    adam_vgg19_train_1e3_accuracy = [0.6415553, 0.7922236, 0.8250304, 0.8821385, 0.8566221, 0.8833536, 0.9149453, 0.8894289, 0.90279466, 0.9258809, 0.9222357, 0.9258809, 0.9368165, 0.93195623, 0.9258809, 0.8942892, 0.92952615, 0.9368165, 0.94046175, 0.9489672]

    #vgg19 0.01
    adam_vgg19_val_1e2_loss = [0.5596203804016113, 0.5481629967689514, 0.5956268310546875, 0.3192550241947174, 0.29651930928230286, 0.7962119579315186, 0.1320134699344635, 0.3312201499938965, 0.12729407846927643, 0.36250752210617065, 0.803580641746521, 0.5686818361282349, 0.4556708037853241, 0.6548165082931519, 0.3151586353778839, 1.2871558666229248, 0.7400950789451599, 0.16228562593460083, 0.07971344888210297, 0.08969095349311829]
    adam_vgg19_val_1e2_accuracy = [0.75, 0.6625000238418579, 0.625, 0.887499988079071, 0.8374999761581421, 0.5375000238418579, 0.875, 0.9125000238418579, 0.8999999761581421, 0.887499988079071, 0.699999988079071, 0.7124999761581421, 0.875, 0.5, 0.875, 0.800000011920929, 0.8500000238418579, 0.925000011920929, 0.887499988079071, 0.887499988079071]

    adam_vgg19_train_1e2_loss = [6.445801785910549, 0.5790995366663139, 0.5083103846229007, 0.4065816777137041, 0.35718655622454487, 0.3255873183252455, 0.39938716686365816, 0.30045173915594203, 0.28221216278641803, 0.29467495978964925, 0.5311984316619025, 0.522615489555593, 0.35394479727252576, 0.570649333560742, 0.4396482555790921, 0.49241110323990533, 0.2994845656549076, 0.3442375540443643, 0.2974016376808376, 0.2963993220091011]
    adam_vgg19_train_1e2_accuracy = [0.52490884, 0.7241798, 0.7691373, 0.8469016, 0.8529769, 0.8675577, 0.81166464, 0.8894289, 0.90157956, 0.890644, 0.79586875, 0.7290401, 0.82867557, 0.77521265, 0.799514, 0.7654921, 0.8979344, 0.8578372, 0.88092345, 0.8784933]

    # ---------------------- SGD -------------------------

    # mobnet 0.001
    sgd_mobile_net_val_1e3_loss = [0.6440691947937012, 0.5417631268501282, 0.31834685802459717, 0.2965807020664215, 0.08907373994588852, 0.09407401084899902, 0.17417281866073608, 0.43657609820365906, 0.3849526047706604, 0.3548003137111664, 0.3583963215351105, 0.3469548225402832, 0.631536066532135, 0.4758070111274719, 0.351920485496521, 0.3137543499469757, 0.14494971930980682, 0.6328288316726685, 0.7387278079986572, 0.6688497066497803]
    sgd_mobile_net_val_1e3_accuracy = [0.800000011920929, 0.762499988079071, 0.8500000238418579, 0.8500000238418579, 0.824999988079071, 0.8374999761581421, 0.8125, 0.8500000238418579, 0.824999988079071, 0.75, 0.8125, 0.8125, 0.8374999761581421, 0.7875000238418579, 0.800000011920929, 0.8374999761581421, 0.7749999761581421, 0.8500000238418579, 0.8125, 0.762499988079071]

    sgd_mobile_net_train_1e3_loss = [0.5888099002664318, 0.4383898650817709, 0.39544002481827906, 0.34049486815205177, 0.3129232717838641, 0.2985476180186828, 0.2824477816627498, 0.35662941896738, 0.3253377825383862, 0.24577223669787368, 0.27059931464647174, 0.22902182539424804, 0.21311865426500398, 0.32900399096391614, 0.25274913800588356, 0.2010076181173216, 0.1945386271683877, 0.22449856728568837, 0.20411975784985437, 0.21982254871005422]
    sgd_mobile_net_train_1e3_accuracy = [0.68043745, 0.80801946, 0.81166464, 0.85540706, 0.86026734, 0.8675577, 0.88456863, 0.85419196, 0.86026734, 0.8942892, 0.88092345, 0.90279466, 0.91859055, 0.85540706, 0.8930741, 0.9161604, 0.927096, 0.90157956, 0.91373026, 0.9210206]


    # mobnet 0.01
    sgd_mobile_net_val_1e2_loss = [0.44632387161254883, 0.443223774433136, 0.6080226302146912, 0.24263574182987213, 0.349161297082901, 0.22206376492977142, 0.1005411371588707, 0.667475700378418, 0.3758392930030823, 0.5540684461593628, 0.5093559622764587, 0.639319896697998, 0.35286015272140503, 0.23195840418338776, 0.3165910243988037, 0.6638514995574951, 0.7353224754333496, 0.7047879695892334, 0.9694844484329224, 0.5445192456245422]
    sgd_mobile_net_val_1e2_accuracy = [0.737500011920929, 0.7749999761581421, 0.8125, 0.862500011920929, 0.7749999761581421, 0.7875000238418579, 0.800000011920929, 0.699999988079071, 0.8500000238418579, 0.800000011920929, 0.7749999761581421, 0.7749999761581421, 0.7875000238418579, 0.8374999761581421, 0.800000011920929, 0.762499988079071, 0.862500011920929, 0.8374999761581421, 0.7749999761581421, 0.824999988079071]

    sgd_mobile_net_train_1e2_loss = [0.833322081719395, 0.43400112594752815, 0.3706011960276132, 0.3708849892581509, 0.37609333951447166, 0.31902613924456397, 0.29390631853302945, 0.320057268601866, 0.27218808228306174, 0.2456163366711082, 0.23404258600054795, 0.227673166667995, 0.2718277796288162, 0.24183641555170243, 0.2396335312474252, 0.2483873065484106, 0.23428118744930693, 0.1933787891534887, 0.20413140209339659, 0.20354059827429122]
    sgd_mobile_net_train_1e2_accuracy = [0.61725396, 0.79100853, 0.83718103, 0.84325635, 0.8311057, 0.872418, 0.88456863, 0.86026734, 0.88456863, 0.90279466, 0.9040097, 0.91251516, 0.8833536, 0.8979344, 0.90157956, 0.8821385, 0.90157956, 0.92952615, 0.9210206, 0.9161604]


    # vgg 0.001
    sgd_vgg19_val_1e3_loss = [0.6584795713424683, 0.6252034902572632, 0.67424076795578, 0.6083027124404907, 0.4581366777420044, 0.6102461814880371, 0.4076412320137024, 0.5025224685668945, 0.4800126552581787, 0.47568511962890625, 0.46995049715042114, 0.3487463593482971, 0.5946827530860901, 0.4685709476470947, 0.3657393157482147, 0.3537083864212036, 0.3347330093383789, 0.28428012132644653, 0.6341546177864075, 0.5315497517585754]
    sgd_vgg19_val_1e3_accuracy = [0.675000011920929, 0.5625, 0.7250000238418579, 0.6625000238418579, 0.824999988079071, 0.7875000238418579, 0.800000011920929, 0.637499988079071, 0.7250000238418579, 0.637499988079071, 0.887499988079071, 0.887499988079071, 0.7124999761581421, 0.75, 0.862500011920929, 0.8374999761581421, 0.8999999761581421, 0.824999988079071, 0.9125000238418579, 0.8374999761581421]

    sgd_vgg19_train_1e3_loss = [0.6824240114535335, 0.6268449016503835, 0.6126575790083133, 0.6177508945815618, 0.5534446596519915, 0.5266053876856552, 0.47559331731366067, 0.4559935937892478, 0.488904062047474, 0.4526504138405384, 0.4252600251795455, 0.5103802276121165, 0.38118511985168135, 0.3881228116419017, 0.3711213871818556, 0.3244401939656871, 0.36864413232299276, 0.30547209781111373, 0.3065044592029414, 0.3839826689962918]
    sgd_vgg19_train_1e3_accuracy = [0.55893075, 0.6524909, 0.6622114, 0.63912517, 0.71202916, 0.7253949, 0.78128797, 0.78128797, 0.7557716, 0.7800729, 0.8347509, 0.74848115, 0.835966, 0.82867557, 0.8420413, 0.86998785, 0.83353585, 0.85905224, 0.87120295, 0.8262454]


    # vgg 0.01
    sgd_vgg19_val_1e2_loss = [0.7226134538650513, 0.6876991987228394, 0.5830408930778503, 0.6776331663131714, 0.722055196762085, 0.7002894878387451, 0.7904285788536072, 0.6969286203384399, 0.45886021852493286, 0.7461180686950684, 0.48648732900619507, 0.6854784488677979, 0.6867141723632812, 0.6972213983535767, 0.6916879415512085, 0.6987737417221069, 0.6986601948738098, 0.6886374950408936, 0.69581538438797, 0.6944347023963928]
    sgd_vgg19_val_1e2_accuracy = [0.5, 0.550000011920929, 0.6875, 0.5, 0.512499988079071, 0.5249999761581421, 0.762499988079071, 0.737500011920929, 0.762499988079071, 0.6000000238418579, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    sgd_vgg19_train_1e2_loss = [0.7389469040265785, 0.6759713069781353, 0.6775852233207675, 0.6513760651519492, 0.6540569258199718, 0.6533323750895689, 0.5777759185176284, 0.5648773227833022, 0.5603232343532915, 0.47596998920127803, 0.5399075314891439, 0.758910795608094, 0.6945236331057867, 0.6939548126547479, 0.6937799652900325, 0.6937154593038964, 0.6939004548120324, 0.6935517484478354, 0.693465287384112, 0.6940491342167953]
    sgd_vgg19_train_1e2_accuracy = [0.5127582, 0.57958686, 0.59538275, 0.65735114, 0.636695, 0.5832321, 0.7144593, 0.7071689, 0.7241798, 0.78493315, 0.73754555, 0.5066829, 0.5066829, 0.50060755, 0.5066829, 0.490887, 0.5066829, 0.5066829, 0.5066829, 0.4799514]


    font = {'family' : 'normal',
            'size'   : 25}
    plt.rc('font', **font)
    '''


    # LOSS
    # plt.plot(adam_mobile_net_train_1e1_loss, color="red", linestyle='dashdot', linewidth=4)
    # plt.plot(adam_mobile_net_train_1e2_loss, color="green", linestyle='dotted', linewidth=4)
    # plt.plot(adam_mobile_net_train_1e3_loss, color="blue", linestyle='dashed', linewidth=4)

    # DONT PLOT VALIDATION LOSS BECAUSE IT'S NOT THAT IMPORTANT AND JUMPS AROUND?
    # plt.plot(adam_mobile_net_val_1e1_loss, color="red", linestyle='dashdot', linewidth=4)
    # plt.plot(adam_mobile_net_val_1e2_loss, color="green", linestyle='dotted', linewidth=4)
    # plt.plot(adam_mobile_net_val_1e3_loss, color="blue", linestyle='dashed', linewidth=4)

    # DONT PLOT TRAIN PRECISION BECAUSE IT STAYS KINDA THE SAME? :(
    # plt.plot(adam_mobile_net_train_1e1_precision, color="red", linestyle='dashdot', linewidth=4)
    # plt.plot(adam_mobile_net_train_1e2_precision, color="green", linestyle='dotted', linewidth=4)
    # plt.plot(adam_mobile_net_train_1e3_precision, color="blue", linestyle='dashed', linewidth=4)

    # ACCURACY
    # plt.plot(adam_mobile_net_train_1e1_accuracy, color="red", linestyle='dashdot', linewidth=4)
    # plt.plot(adam_mobile_net_train_1e2_accuracy, color="green", linestyle='dotted', linewidth=4)
    # plt.plot(adam_mobile_net_train_1e3_accuracy, color="blue", linestyle='dashed', linewidth=4)

    # plt.plot(adam_mobile_net_val_1e1_accuracy, color="red", linestyle='dashdot', linewidth=4)
    # plt.plot(adam_mobile_net_val_1e2_accuracy, color="green", linestyle='dotted', linewidth=4)
    # plt.plot(adam_mobile_net_val_1e3_accuracy, color="blue", linestyle='dashed', linewidth=4)

    # plt.plot(adam_vgg19_val_1e2_accuracy, color="green", linestyle='dotted', linewidth=4)
    # plt.plot(adam_vgg19_val_1e3_accuracy, color="blue", linestyle='dashed', linewidth=4)

    # axes = plt.gca()
    # axes.set_ylim([0,1])

    # plt.xlabel('EPOCH')
    # plt.ylabel('ACCURACY')
    # plt.xticks(range(EPOCHS))
    # plt.legend(['ϵ = 0.01', 'ϵ = 0.001'], loc='lower right')
    # plt.show()

    # plt.close()

    '''

    model_getters = ["256x256x256x256x1_SGD_VGG19", get_vgg19_net_model(), None] # ["256x256x256x256x1_SGD_MOBNET", get_mobile_net_model(), None] # 

    LEARNING_RATES = [1e-2]  # , 1e-2, 1e-1]

    getter = model_getters
    print(model_getters)
    model = Sequential()
    model.name = getter[0]
    model.add(getter[1])
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))    # 1 when binary, 2 when categorical

    for lr in LEARNING_RATES:
        print("LEARN ME BRO")

        train_generator = get_train_data(getter[2])
        validation_generator = get_validation_data(getter[2])

        print(train_generator.n)
        print(validation_generator.n)

        model.compile(loss='binary_crossentropy',
                      # optimizer=Adam(learning_rate=lr),
                      optimizer=SGD(learning_rate=lr, momentum=0.9, nesterov=False),
                      metrics=MODEL_METRICS)

        history = model.fit_generator(train_generator,
                                      epochs=EPOCHS,
                                      steps_per_epoch=train_generator.n / BATCH_SIZE,
                                      validation_data=validation_generator,
                                      validation_steps=validation_generator.n / BATCH_SIZE)

        name = str(model.name) + "_" + str(lr) + ".txt"
        f = open(name, "w")
        f.write(str(history.history))
        f.close()

        # plt.plot(history.history['loss'])
        # plt.title('loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.xticks(range(EPOCHS))
        # plt.legend(['loss'], loc='upper left')
        # plt.savefig(str(model.name) + "_" + str(lr) +
        #             "_loss.png", bbox_inches='tight')
        # plt.close()

        # plt.plot(history.history['binary_accuracy'], label="accuracy")
        # plt.plot(history.history['precision_1'], label="precision")
        # plt.xticks(range(EPOCHS))
        # plt.legend()
        # plt.savefig(str(model.name) + "_" + str(lr) +
        #             "_acc.png", bbox_inches='tight')
        # plt.close()

        # plt.plot(history.history['val_binary_accuracy'], label="val_accuracy")
        # plt.plot(history.history['val_precision_1'], label="val_precision")
        # plt.xticks(range(EPOCHS))
        # plt.legend()
        # plt.savefig(str(model.name) + "_" + str(lr) +
        #             "_val.png", bbox_inches='tight')
        # plt.close()

        model.save(model.name + "_" + str(lr) + ".h5")

        # plot_training_history(history)
    '''
    # JESUS MODEL TAKE CARE OF IT
    model = load_model('mobilenet_rescale.h5', compile=False)
    # model = load_model('256x256x256x256x1_ADAM_MOBNET_0.001.h5', compile=False)
    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_datagen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=16,
        class_mode=CLASS_MODE,
        shuffle=True)

    # predictions = model.predict_generator(test_generator)
    bla = None
    pred = None
    for batch in test_datagen:
        results = predict(model, batch[0])

        fig, axs = plt.subplots(4,4, facecolor='w', edgecolor='k')
        fig.subplots_adjust(wspace=.2)
        axs = axs.ravel()
        for i in range(16):
            prob = round(1 - results[i][0], 2)
            if prob > 0.5:
                axs[i].set_title("Ball " + str(prob))
            else:
                axs[i].set_title("None " + str(prob))

            axs[i].imshow(batch[0][i])
            axs[i].tick_params(
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,
                labelbottom=False,
                labelleft=False) # labels along the bottom edge are off
        font = {'family' : 'normal',
                'size'   : 25}
        plt.rc('font', **font)
        plt.show()

        # for img, res in zip(batch[0], results):

        # fig, ax = plt.subplots(4, 4)
        # idx = 0
        # for i in range(0, 4):
        #     for j in range(0, 4):
        #         ax[i, j].imshow(batch[0][idx])
        #         # ax[i, j]..xlabel(str(round(results[0][idx], 2)))
        #         idx += 1
        # plt.show()

    # 16 images in batch[0], 16 results in results
