from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Imagens de teste e treinamento da biblioteca
(x_treino, y_treino), (x_teste, y_teste) = load_data()

# Imagens 28x28
altura_img = x_treino.shape[1]
largura_img = x_treino.shape[1]
numero_canais = 1

x_treino = np.expand_dims(x_treino, -1)
x_teste = np.expand_dims(x_teste, -1)

# Normalização dos dados (valores entre 0 e 1)
x_treino = x_treino.astype('float32') / 255.
x_teste = x_teste.astype('float32') / 255.

# Categorização das classes - vetor para binário
num_classes = 10
y_treino = to_categorical(y_treino, num_classes)
y_teste = to_categorical(y_teste, num_classes)

print(x_treino.shape[0], "dados de treinamento")
print(x_teste.shape[0], "dados de teste")

# Separação de dados de treino para validação
indices = 0
# Pegando indices aleatorios no conjunto de treinamento
for _ in range(5):
    indices = np.random.permutation(len(x_treino))

x_treino = x_treino[indices]
y_treino = y_treino[indices]

# perc_validacao = percentual de dados que serão utilizados na validação
perc_validacao = .1
val_count = int(perc_validacao * len(x_treino))

# Separacao do conjunto para validacão e treinamento
x_validacao = x_treino[:val_count, :]
y_validacao = y_treino[:val_count, :]
x_treino = x_treino[val_count:, :]
y_treino = y_treino[val_count:, :]


# Função que constroi o modelo da rede
def construir_modelo():
    modelo = Sequential()
    # Camadas convolucionais
    modelo.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                      input_shape=(altura_img, largura_img, numero_canais)))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    # Saida
    modelo.add(Dense(num_classes, activation='softmax'))
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


def graficos(historico):
    loss_vals = historico['loss']
    val_loss_vals = historico['val_loss']
    epochs = range(1, len(historico['accuracy']) + 1)

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    # Perdas
    ax[0].plot(epochs, loss_vals, color='navy', marker='o', linestyle=' ', label='Training Loss')
    ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')
    ax[0].grid(True)

    # Acurácias
    acc_vals = historico['accuracy']
    val_acc_vals = historico['val_accuracy']

    ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')
    ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')
    ax[1].grid(True)

    plt.show()
    plt.close()

    # liberar espaço na pilha
    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals


def main():
    modelo = construir_modelo()
    print(modelo.summary())
    resultados = modelo.fit(x_treino, y_treino,
                            epochs=15, batch_size=64,
                            validation_data=(x_validacao, y_validacao))
    graficos(resultados.history)


if __name__ == '__main__':
    main()
