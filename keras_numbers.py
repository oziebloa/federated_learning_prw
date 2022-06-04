import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
# Dane z kerasa nie wymagają podziału na dane treningowe oraz testowe,
# ponieważ domyślnie ten zestaw danych został już podzielony
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# W tym zestawie danych x_train, x_test to zestawy obrazów,
# a y_train oraz y_test to odpowiednio ich klasyfikacje, czyli jaką cyfrę dany obraz reprezentuje
# x_train, oraz x_test są trójwymiarowymi tabelami odpowiednio 60000x28x28 i 10000x28x28, aby przejść dalej

# W następnym kroku normalizuje zestaw danych, czyli zamieniam do tej pory obrazy w odcieniach szarości, czyli takiego
# gdzie każdy bit może przybrać wartość od 0 do 255, na taki, gdzie każdy bit może przybrać tylko wartość 0 lub 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

### TWORZENIE ORAZ TRENOWANIE MODELU ###

# Mając przygotowany zbiór danych przechodzę do tworzenia modelu, typ sequential oznacza liniowo ułożone warstwy
model = tf.keras.models.Sequential()
# Zaczynam od stworzenia warstwy spłaszczającej obraz, czyli zamieniającej macierz
# 28x28 na pojedynczy ciąg 784 znaków
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Tworzę pierwszą ukrytą warstwę
model.add(tf.keras.layers.Dense(input_dim=x_train.shape[1], units=256, kernel_initializer='uniform', activation='relu'))
# Dodaje output layer
model.add(tf.keras.layers.Dense(units=10, kernel_initializer='uniform', activation='softmax'))
# Kompiluje model co kończy moją interakcjęz narzedziem keras
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Pierwszy model dopasowywuje domyślną metodą biblioteki keras (nie jest to tyle konkretny algorytm
#co dopasowanie bazujące na stworzonym modelu korzystając z optymizra przekazanego w kompilacji)
model.fit(x_train, y_train, epochs = 10)



