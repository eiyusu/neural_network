from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Variáveis globais
altura_img = 28
largura_img = 28
numero_canais = 1
num_classes = 10

# Casos de avaliação
casos = ['uma_conv_norm_dez',
         'uma_conv_nao_norm_dez',
         'uma_conv_norm_trinta',
         'uma_conv_nao_norm_trinta',
         'duas_conv_norm_dez',
         'duas_conv_nao_norm_dez',
         'duas_conv_norm_trinta',
         'duas_conv_nao_norm_trinta',
         'tres_conv_norm_dez',
         'tres_conv_nao_norm_dez',
         'tres_conv_norm_trinta',
         'tres_conv_nao_norm_trinta']

# Camadas convolucionais, Normalização, Percentual de dados para Validação
valores_casos = [[0, True, .1],
                 [0, False, .1],
                 [0, True, .3],
                 [0, False, .3],
                 [1, True, .1],
                 [1, False, .1],
                 [1, True, .3],
                 [1, False, .3],
                 [2, True, .1],
                 [2, False, .1],
                 [2, True, .3],
                 [2, False, .3]]


# Função que gera os dados de treinamento, com uma parte para validação, e teste
def dados_treinamento(normalizacao, perc_validacao):
    # Imagens de teste e treinamento da biblioteca
    (x_treino, y_treino), (x_teste, y_teste) = load_data()

    x_treino = np.expand_dims(x_treino, -1)
    x_teste = np.expand_dims(x_teste, -1)

    # Normalização dos dados (valores entre 0 e 1)
    if normalizacao:
        x_treino = x_treino.astype('float32') / 255.
        x_teste = x_teste.astype('float32') / 255.

    # Categorização das classes - vetor para binário
    y_treino = to_categorical(y_treino, num_classes)
    y_teste = to_categorical(y_teste, num_classes)

    # Separação de dados de treino para validação
    indices = 0
    # Pegando indices aleatorios no conjunto de treinamento
    for _ in range(5):
        indices = np.random.permutation(len(x_treino))

    x_treino = x_treino[indices]
    y_treino = y_treino[indices]

    # perc_validacao = percentual de dados que serão utilizados na validação
    val_count = int(perc_validacao * len(x_treino))

    # Separacao do conjunto para validacão e treinamento
    x_validacao = x_treino[:val_count, :]
    y_validacao = y_treino[:val_count, :]
    x_treino = x_treino[val_count:, :]
    y_treino = y_treino[val_count:, :]

    return [x_validacao, y_validacao, x_treino, y_treino, x_teste, y_teste]


# Função que constroi o modelo da rede
def construir_modelo(conv_num):
    modelo = Sequential()
    # Camadas convolucionais
    modelo.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                      input_shape=(altura_img, largura_img, numero_canais)))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(0, conv_num):
        modelo.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    # Saida
    modelo.add(Dense(num_classes, activation='softmax'))
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


# Função que gera gráficos de perda e acurácia do modelo por epoca
def graficos(historico, caso):
    loss_vals = historico['loss']
    val_loss_vals = historico['val_loss']
    epochs = range(1, len(historico['accuracy']) + 1)

    # Perdas
    plt.plot(epochs, loss_vals, color='navy', marker='o', linestyle=' ', label='Perda de Treinamento')
    plt.plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Perda de Validação')
    plt.title('Perda de Treinamento & Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend(loc='best')
    plt.grid(True)
    titulo_perda = 'perda_' + caso
    plt.savefig(titulo_perda)
    plt.close()

    # Acurácias
    acc_vals = historico['accuracy']
    val_acc_vals = historico['val_accuracy']
    plt.plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Acurácia de Treinamento')
    plt.plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Acurácia de Validação')
    plt.title('Acurácia de Treinamento & Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend(loc='best')
    plt.grid(True)
    titulo_acuracia = 'acuracia_' + caso
    plt.savefig(titulo_acuracia)
    plt.close()

    # liberar espaço na pilha
    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals


# Função auxiliar que salva em um arquivo o resultado de perda e acurácia para os dados de teste
def salvar_teste(caso, perda, acuracia):
    caso = caso + '_teste.txt'
    f = open(caso, 'w+')
    f.write('Acuracia: %f\n' % acuracia)
    f.write('Perda: %f\n' % perda)
    f.close()


# Função que roda teste utilizando o modelo, feita para aceitar diferentes casos
def rodar_teste(caso, quantidade_convolucao, normalizacao, perc_validacao):
    modelo = construir_modelo(quantidade_convolucao)
    [x_validacao, y_validacao, x_treino, y_treino, x_teste, y_teste] = dados_treinamento(normalizacao, perc_validacao)
    print(modelo.summary())
    resultados = modelo.fit(x_treino, y_treino,
                            epochs=15, batch_size=64,
                            validation_data=(x_validacao, y_validacao))
    graficos(resultados.history, caso)
    perda_teste, acuracia_teste = modelo.evaluate(x_teste, y_teste, batch_size=64)
    salvar_teste(caso, perda_teste, acuracia_teste)


def main():
    for i in range(0, len(casos)):
        rodar_teste(casos[i], valores_casos[i][0], valores_casos[i][1], valores_casos[i][2])


if __name__ == '__main__':
    main()
