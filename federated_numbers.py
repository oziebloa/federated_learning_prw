from __future__ import absolute_import, division, print_function


import collections
from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff


##############################################################################################################################
################################### TWORZENIE SFEDEROWANEGO (FEDERATED) ZESTAWU DANYCH #######################################
##############################################################################################################################

#Zmienne wykorzystywane do przetransformowania i spłaszczenia obrazów ze zbioru danych MNIST w X i Y wykorzystywane dalej w modelu kerasowym
NUM_EPOCHS = 100
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

nest = tf.contrib.framework.nest
#Ustawiam numpy random seed do startowania z zera
np.random.seed(0)
tf.compat.v1.enable_v2_behavior()
#Zamiast zwykłych danych MNIST wczytuję wersję federated, czyli zebrany od wielu użytkowników
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
#Zaczynam tworzenie klientów. Do symulacji wykorzystuję stałą liczbę klientów, natomiast z reguły liczba klientów nie będzie z góry określona
NUM_CLIENTS = 5
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)



#Funkcja wykorzystywana to spłaszczenia 28x28 datasetu w wektor o 784 elementach
def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)

#Funkcja tworząca sfederowane zestawy danych z podziałem na klientów
def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]




##############################################################################################################################
################################### TWORZENIE MODELU KERAS ###################################################################
##############################################################################################################################

#Funkcja tworząca model ze spłaszczonego (do tablicy  wektorów po 784 elementów) zbioru danych
def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
    #Definicja funkcji straty
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred))
    #Wybór opcji kompilacji modelu
    model.compile(
        loss=loss_fn,
        optimizer=gradient_descent.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

#Tworzę model
def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

#Sprawdzam jakość modelu
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
str(iterative_process.initialize.type_signature)
state = iterative_process.initialize()
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))
for round_num in range(2, 101):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))