import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tqdm
import skimage.io
import glob
from datetime import datetime

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


class History_Tensor(tf.keras.callbacks.Callback):
    # callback function and class adjustment for calculating test accuracy and loss

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.test_data)
        if 'test_loss' not in logs:
            # logs['test_loss'] = []
            # logs['test_loss'] = None
            logs['test_loss'] = np.nan
        else:
            # logs['test_loss'].append(loss)
            logs['test_loss'] = loss
        if 'test_accuracy' not in logs:
            # logs['test_accuracy'] = []
            # logs['test_accuracy'] = None
            logs['test_accuracy'] = np.nan
        else:
            logs['test_accuracy'] = acc


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)  # normalizes feature vectors
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class Contrastive:

    def __init__(self):
        self.p_project = '/home/edoli/PycharmProjects/Contrastive-Prediction'
        self.p_resource = self.p_project + '/resource/C-NMC_Leukemia'

    def train_baseline_model(self, X, y, X_val, y_val, input_shape, num_epochs, batch_size, df_results):
        # Model 1: Baseline Inception V3 #
        s_model = 'Inception_V3'
        num_classes = 2
        b_contrastive = False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        train_datagen = ImageDataGenerator(horizontal_flip=True,
                                           vertical_flip=True,
                                           zoom_range=0.2,
                                           preprocessing_function=preprocess_input)
        # train_datagen.fit(X)
        train_datagen.fit(X_train)

        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        valid_datagen.fit(X_val)

        # creates model with pre trained imagenet weights
        incep_v3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(input_shape, input_shape, 3))

        incep_v3.summary()  # model summary

        # does not train all layers therefore uses same weights as the given model
        for layers in incep_v3.layers:
            layers.trainable = False

        x = Flatten()(incep_v3.output)  # flattens layer

        fcc_layer_1 = Dense(units=1024, activation='relu')(x)  # fcc and output layer
        dropout_1 = Dropout(0.3)(fcc_layer_1)

        fcc_layer_2 = Dense(units=512, activation='relu')(dropout_1)
        dropout_2 = Dropout(0.3)(fcc_layer_2)

        final_layer = Dense(units=1, activation='sigmoid')(dropout_2)

        model = Model(inputs=incep_v3.input, outputs=final_layer)  # creates final model

        model.summary()  # model summary

        # _metrics = ['accuracy']

        _metrics = [keras.metrics.Accuracy()]

        # _metrics = [keras.metrics.Accuracy(),
        #             keras.metrics.AUC(),
        #             keras.metrics.Recall(),
        #             keras.metrics.Precision(),
        #             tfa.metrics.F1Score(num_classes=num_classes, average='weighted')
        #             ]

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=_metrics)

        l_callbacks = self.set_callbacks(b_contrastive, X_test)  # callbacks

        # history = model.fit(train_datagen.flow(X, y, batch_size=batch_size),
        #                     validation_data=(X_val, y_val),
        #                     epochs=num_epochs,
        #                     verbose=1,
        #                     callbacks=l_callbacks)

        history = model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                            validation_data=(X_val, y_val),
                            epochs=num_epochs,
                            verbose=1,
                            callbacks=l_callbacks)

        self.plot_acc_loss(history, s_model, b_contrastive=False)

        scores = model.evaluate(X_test, y_test)

        loss = scores[0]
        accuracy = scores[1]

        d_results = {'Model': s_model, 'Accuracy': accuracy, 'Loss': loss}
        df_results = df_results.append(d_results, ignore_index=True)

        return df_results

    def init_model(self, input_shape, num_classes=2):
        curr_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling="avg",
            classes=num_classes,
            classifier_activation="softmax",
        )

        # curr_model = tf.keras.applications.ResNet50V2(
        #     include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        # )

        # curr_model = tf.keras.applications.Xception(include_top=False,
        #                                             weights=None,
        #                                             input_shape=input_shape,
        #                                             pooling="avg",
        #                                             classes=num_classes,
        #                                             classifier_activation="softmax"
        #                                            )

        return curr_model

    def get_augmentation(self, X_train):
        # data_augmentation = keras.Sequential(
        #     [
        #         layers.Normalization(),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomRotation(0.02),
        #         layers.RandomWidth(0.2),
        #         layers.RandomHeight(0.2),
        #     ]
        # )

        data_augmentation = keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.Normalization(),
                tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.02),
                tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
                tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
            ]
        )

        # data_augmentation = keras.Sequential(
        #     [
        #         layers.Normalization(),
        #         layers.RandomCrop(128, 128),
        #         layers.RandomZoom(0.5, 0.2),
        #         layers.RandomContrast(0.2),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomRotation(0.02),
        #         layers.RandomWidth(0.2),
        #         layers.RandomHeight(0.2)
        #     ]
        # )

        data_augmentation.layers[0].adapt(X_train)  # sets the state of the normalization layer

        return data_augmentation

    def create_encoder(self, X_train, input_shape):
        shape = (input_shape, input_shape, 3)
        curr_model = self.init_model(shape)
        data_augmentation = self.get_augmentation(X_train)
        inputs = keras.Input(shape=shape)
        augmented = data_augmentation(inputs)
        outputs = curr_model(augmented)
        model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive-encoder")
        return model

    def create_classifier(self, encoder, input_shape, trainable=True):
        shape = (input_shape, input_shape, 3)
        learning_rate = 0.001
        hidden_units = 512
        num_classes = 2
        dropout_rate = 0.5

        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = keras.Input(shape=shape)
        features = encoder(inputs)
        features = layers.Dropout(dropout_rate)(features)
        features = layers.Dense(hidden_units, activation="relu")(features)
        features = layers.Dropout(dropout_rate)(features)
        outputs = layers.Dense(num_classes, activation="softmax")(features)

        model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive-clf")

        # _metrics = [keras.metrics.SparseCategoricalAccuracy(),
        #             keras.metrics.Recall(),
        #             keras.metrics.Precision(),
        #             tfa.metrics.F1Score(num_classes=num_classes, average='weighted'),
        #             keras.metrics.AUC()
        #             ]

        _metrics = [keras.metrics.SparseCategoricalAccuracy()]

        # _loss = 'binary_crossentropy'

        _loss = keras.losses.SparseCategoricalCrossentropy()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=_loss,
            metrics=_metrics
        )

        return model

    def plot_acc_loss(self, history, m_name, b_contrastive):
        # function displays loss and accuracy graphs of the models.
        p_plots = '/home/edoli/PycharmProjects/Contrastive-Prediction'

        if b_contrastive:
            plt.plot(history.history['sparse_categorical_accuracy'])
            # plt.plot(history.history['val_sparse_categorical_accuracy'])
        else:
            plt.plot(history.history['accuracy'])
            # plt.plot(history.history['val_accuracy'])

        # plt.plot(history.history['test_accuracy'])

        # accuracy plot
        plt.title('Model ' + m_name + ' Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        # plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        p_save_acc = p_plots + '/' + m_name + '_acc.png'
        plt.savefig(p_save_acc)
        plt.show()
        plt.clf()

        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])

        # plt.plot(history.history['test_loss'])

        # loss plot
        plt.title('Model ' + m_name + ' Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        # plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        p_save_loss = p_plots + '/' + m_name + '_loss.png'
        plt.savefig(p_save_loss)
        plt.show()
        plt.clf()

    def add_projection_head(self, encoder, input_shape):
        shape = (input_shape, input_shape, 3)
        projection_units = 128
        inputs = keras.Input(shape=shape)
        features = encoder(inputs)
        outputs = layers.Dense(projection_units, activation="relu")(features)
        model = keras.Model(
            inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
        )
        return model

    def load_datasets(self):
        train_dataset_0_all = glob.glob(self.p_resource + '/training_data/fold_0/all/*.bmp')
        train_dataset_0_hem = glob.glob(self.p_resource + '/training_data/fold_0/hem/*.bmp')
        train_dataset_1_all = glob.glob(self.p_resource + '/training_data/fold_1/all/*.bmp')
        train_dataset_1_hem = glob.glob(self.p_resource + '/training_data/fold_1/hem/*.bmp')
        train_dataset_2_all = glob.glob(self.p_resource + '/training_data/fold_2/all/*.bmp')
        train_dataset_2_hem = glob.glob(self.p_resource + '/training_data/fold_2/hem/*.bmp')

        A, H = list(), list()
        A.extend(train_dataset_0_all)
        A.extend(train_dataset_1_all)
        A.extend(train_dataset_2_all)
        H.extend(train_dataset_0_hem)
        H.extend(train_dataset_1_hem)
        H.extend(train_dataset_2_hem)

        return A, H

    def preprocess(self, A, H, input_shape, b_sample):
        A = np.array(A)
        H = np.array(H)

        Image = []
        Label = []

        if b_sample:
            length_a = 1000
            length_h = 500
        else:
            length_a = len(A)  # 7272
            length_h = len(H)  # 3389

        for i in tqdm(range(0, length_a)):
            img = imread(A[i])
            img = resize(img, (input_shape, input_shape))
            Image.append(img)
            Label.append(1)

        for i in tqdm(range(0, length_h)):
            img = imread(H[i])
            img = resize(img, (input_shape, input_shape))
            Image.append(img)
            Label.append(0)

        Image = np.array(Image)
        Label = np.array(Label)

        Image, Label = shuffle(Image, Label, random_state=42)  # shuffle

        # fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))  # plots examples
        # for i in tqdm(range(0, 5)):
        #     rand = np.random.randint(len(Image))
        #     ax[i].imshow(Image[rand])
        #     ax[i].axis('off')
        #     a = Label[rand]
        #     if a == 1:
        #         ax[i].set_title('Diseased')
        #     else:
        #         ax[i].set_title('Non_Diseased')

        return Image, Label

    def load_validation_set(self, input_shape):
        valid_data = pd.read_csv(self.p_resource + '/validation_data/C-NMC_test_prelim_phase_data_labels.csv')
        valid_data.head()

        X_val = []  # loads image and storing it numpy array.

        for image_name in valid_data.new_names:
            img = imread(self.p_resource + '/validation_data/C-NMC_test_prelim_phase_data/' + image_name)
            img = resize(img, (input_shape, input_shape))
            X_val.append(img)

        X_val = np.array(X_val)  # converts to array

        y_val = valid_data.labels.values  # stores target values

        return X_val, y_val

    def decay(self, epoch):
        if epoch < 3:
            print('Learning Rate for Epoch: 1e-6')
            return 1e-6
        elif 3 <= epoch < 10:
            print('Learning Rate for Epoch: 1e-8')
            return 1e-8
        elif 10 <= epoch < 20:
            print('Learning Rate for Epoch: 1e-10')
            return 1e-10
        else:
            print('Learning Rate for Epoch: 1e-12')
            return 1e-12

    def Learning_Tensor(self):
        return tf.keras.callbacks.LearningRateScheduler(self.decay)

    def plateu_and_early_stop(self, params):
        # callback function stops the model from continuing to run and changes learning rate during training.
        lr_factor = 0.1
        # lr_factor = 0.99

        # min_d = 1e-4
        min_d = 0

        stop = tf.keras.callbacks.EarlyStopping(monitor=params, patience=8, verbose=1, min_delta=min_d,
                                                restore_best_weights=True)

        plateu = tf.keras.callbacks.ReduceLROnPlateau(monitor=params, factor=lr_factor, patience=8, verbose=1,
                                                      min_delta=min_d)
        return stop, plateu

    def set_callbacks(self, b_contrastive, test=None):

        if b_contrastive:
            _monitor = 'val_sparse_categorical_accuracy'
        else:
            _monitor = 'val_accuracy'

        early_stopping = EarlyStopping(monitor=_monitor,
                                      mode='max',
                                      patience=15,
                                      verbose=1)

        # p_checkpoint = self.p_project + '/best_weights.hdf5'
        # checkpoint = ModelCheckpoint(p_checkpoint,
        #                              monitor=_monitor,
        #                              mode='max',
        #                              save_best_only=True,
        #                              verbose=1)

        learning_rate = ReduceLROnPlateau(monitor=_monitor,
                                          mode='max',
                                          patience=5,
                                          factor=0.3,
                                          min_delta=0.00001)

        # plotting = tf.keras.callbacks.TensorBoard(self.p_project, update_freq=1)

        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', min_delta=0, patience=8, verbose=1, restore_best_weights=True
        # )

        # learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss', factor=0.1, patience=8, verbose=1,
        #     min_delta=0, cooldown=0, min_lr=0,
        # )

        # learning_rate = self.plateu_and_early_stop("val_loss")

        # checkpoint = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=self.p_project, monitor='val_loss', verbose=1, save_best_only=True,
        #     save_weights_only=False, save_freq=1, options=None,
        # )

        # tensor_history = History_Tensor(test)

        # tensor_lrdecay = self.Learning_Tensor()

        # l_callbacks = [learning_rate, early_stopping]
        # l_callbacks = [plotting, tensor_history, learning_rate]
        # l_callbacks = [learning_rate, tensor_history]
        l_callbacks = [learning_rate]

        return l_callbacks

    def train_contrastive_model(self, X, y, X_val, y_val, input_shape, num_epochs, batch_size, df_results):
        # Model 2: Contrastive Learning #

        # learning_rate = 0.01
        learning_rate = 0.001

        temperature = 0.05
        # temperature = 0.1
        # temperature = 0.2

        b_contrastive = True

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # y_train = np.expand_dims(y_train, axis=1)

        # l_callbacks = self.set_callbacks(b_contrastive, X_test)

        encoder = self.create_encoder(X_train, input_shape)
        encoder.summary()

        print('Training Contrastive Encoder')
        encoder = self.create_encoder(X_train, input_shape)
        encoder_with_projection_head = self.add_projection_head(encoder, input_shape)
        encoder_with_projection_head.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=SupervisedContrastiveLoss(temperature),
        )
        encoder_with_projection_head.summary()

        history = encoder_with_projection_head.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

        print('Training Classifier on top of the Contrastive Encoder')
        classifier = self.create_classifier(encoder, input_shape, trainable=False)
        s_model = 'Contrastive_Inception_V3'

        history = classifier.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

        # curr_steps_per_epoch = X_train.shape[0] // batch_size
        # curr_validation_steps = X_val.shape[0] // batch_size
        # history = classifier.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
        #                          validation_data=(X_val, y_val), callbacks=l_callbacks,
        #                          steps_per_epoch=curr_steps_per_epoch, validation_steps=curr_validation_steps,
        #                          shuffle=True)

        scores = classifier.evaluate(X_test, y_test)

        loss = scores[0]
        accuracy = scores[1]

        # recall = scores[2]
        # precision = scores[3]
        # auc = scores[4]
        # f1 = scores[5]
        # prauc = score[6]
        # print(f'Test accuracy: {round(accuracy * 100, 2)}% \n recall: {round(recall * 100, 2)}% \n precision: {round(precision * 100, 2)}% \n auc: {round(auc * 100, 2)}% \n f1: {round(f1 * 100, 2)}%')

        self.plot_acc_loss(history, s_model, b_contrastive)
        print(f'Model: {s_model}, Test Accuracy: {round(accuracy * 100, 2)}%, Test Loss: {round(loss * 100, 2)}%')

        y_preds = classifier.predict(X_test).reshape(-1, 1)

        # y_probs = classifier.predict_proba(X_test)y_probs[:, 1]

        test_acc = accuracy_score(y_test, y_preds)
        # precision = precision_score(y_test, y_preds)
        # recall = recall_score(y_test, y_preds)
        # f1 = f1_score(y_test, y_preds)
        # fpr, tpr, threshold = roc_curve(y_test, y_preds)
        # auc_score = roc_auc_score(y_test, y_probs)
        # arr_precision, arr_recall, threshold_pr_auc = precision_recall_curve(y_test, y_probs)
        # prauc = auc(arr_recall, arr_precision)

        # d_results = {'Model': s_model, 'Accuracy': accuracy, 'AUC': auc_score, 'Recall': recall, 'Precision': precision,
        #              'F1': f1, 'PRAUC': prauc}

        # d_results = {'Model': s_model, 'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1}

        # d_results = {'Model': s_model, 'Accuracy': accuracy, 'Loss': loss}
        d_results = {'Model': s_model, 'Accuracy': test_acc}
        df_results = df_results.append(d_results, ignore_index=True)

        return df_results


if __name__ == '__main__':
    input_shape = 128
    # input_shape = 256
    # input_shape = 300

    num_epochs = 200
    # num_epochs = 500

    # batch_size = 512
    # batch_size = 256
    # batch_size = 128
    # batch_size = 64
    batch_size = 32

    # b_sample = True
    b_sample = False

    contra = Contrastive()

    A, H = contra.load_datasets()

    X, y = contra.preprocess(A, H, input_shape, b_sample)
    
    X_val, y_val = contra.load_validation_set(input_shape)

    l_cols = ['Model', 'Accuracy']
    # l_cols = ['Model', 'Accuracy', 'Loss']
    # l_cols = ['Model', 'Accuracy', 'Loss', 'AUC', 'Recall', 'Precision', 'F1', 'PRAUC']
    df_results = pd.DataFrame(columns=l_cols)
    s_filename = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    p_output = contra.p_project + '/' + s_filename + '.csv'

    # df_results = contra.train_baseline_model(X, y, X_val, y_val, input_shape, num_epochs, batch_size, df_results)

    df_results = contra.train_contrastive_model(X, y, X_val, y_val, input_shape, num_epochs, batch_size, df_results)

    df_results.to_csv(path_or_buf=p_output, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
