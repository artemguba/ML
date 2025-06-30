import numpy as np
import matplotlib.pyplot as plt
import torch

# компоненты Pytorch для создания и обучения сети
from torch.nn import (Module, Sequential,
                      Linear, ReLU, Sigmoid,
                      MSELoss)
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.optim import Adam

# класс автокодировщика --- сети, которая создаёт на выходе
# копию входа, при этом число нейронов на одном из внутренних слоёв
# меньше, чем на входе и выходе. Она полезна для создания
# сжатых представлений данных.
class Autoencoder(Module):
    def __init__(self):
        # инициализируем родительский класс
        Module.__init__(self)
        # создаём переменные для числа скрытых нейронов
        layer1_num = 128
        layer2_num = 64
        layer3_num = 25
        # создаём сеть-кодировщик.
        # Здесь мы пытаемся сделать сразу целую сеть,
        # а не обучать отдельные пары слоёв
        self.encoder = Sequential(
            Linear(28 * 28, layer1_num),
            ReLU(),
            Linear(layer1_num, layer2_num),
            ReLU(),
            Linear(layer2_num, layer3_num))

        # создаём сеть-декодировщик с аналогичными слоями,
        # расположенными в обратном порядке. Последний
        # слой позволяет ограничить диапазон выходов
        self.decoder = Sequential(
            Linear(layer3_num, layer2_num),
            ReLU(),
            Linear(layer2_num, layer1_num),
            ReLU(),
            Linear(layer1_num, 28*28),
            Sigmoid())

    def forward(self, x):
        # при работе сети пропускаем данные
        # через обе части последовательно
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Гиперпараметры для обучения:
# скорость обучения (learning rate)
LR = 1e-3
WEIGHT_DECAY = 1e-8
# размер одного куска данных
BATCH_SIZE = 64
# число эпох обучения
EPOCHS = 10

# инициализируем модель
model = Autoencoder()

# качество будем оценивать с помощью MSE
loss_function = MSELoss()

# используем оптимизацию методов Адама
optimizer = Adam(model.parameters(),
                 lr = LR,
                 weight_decay = WEIGHT_DECAY)

# Пример автоматически загружает данные из выборки MNIST -
# одного из бенчмарков для отладки распознавателей:
# https://github.com/rois-codh/kmnist
print("Loading the MNIST dataset...")
trainData = MNIST(root="data", train=True, download=True,
    transform=ToTensor())
# загружаем тестовую выборку отдельно от обучающей
testData = MNIST(root="data", train=False, download=True,
    transform=ToTensor())

# Создаём объекты-загрузчики данных (класс DataLoader):
# обучающую выборку важно перемешать, иначе она будет подаваться
# на сеть в том же порядке, в каком хранится, поэтому
# для неё shuffle = True
trainDataLoader = DataLoader(trainData,batch_size = BATCH_SIZE,
                             shuffle = True)

testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

losses = []
train_mse_list = []
print("Net training...")
for epoch in range(EPOCHS):
    total_train_mse = 0
    num_train_samples = 0
    for (image, _) in trainDataLoader:

      # преобразуем данные из изображений (матриц) 28x28
      # в вектора длины 28*28
      image = image.reshape(-1, 28*28)

      # получаем закодированное и декодированное изображение
      # от автокодировщика
      reconstructed = model(image)

      # вычисляем функцию потерь
      loss = loss_function(reconstructed, image)

      # обнуляем градиент
      optimizer.zero_grad()
      # обсчитываем обратное распространение ошибки
      loss.backward()
      # обновляем параметры сети
      optimizer.step()

      # добавляем функцию ошибки в массив-лог
      losses.append(loss.item())

      # вычисляем MSE для текущего батча
      total_train_mse += loss.item() * image.size(0)
      num_train_samples += image.size(0)

    # вычисляем среднее значение MSE по всей обучающей выборке
    average_train_mse = total_train_mse / num_train_samples
    train_mse_list.append(average_train_mse)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Average Train MSE: {average_train_mse}")

# выводим график из 100 последних ошибок
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses[-100:])
plt.show()

# Оценка точности восстановления данных на тестовой выборке
total_mse = 0
num_samples = 0

with torch.no_grad():
    for (image, _) in testDataLoader:
        # преобразуем данные из изображений (матриц) 28x28
        # в вектора длины 28*28
        image = image.reshape(-1, 28*28)

        # получаем закодированное и декодированное изображение
        # от автокодировщика
        reconstructed = model(image)

        # вычисляем MSE для текущего батча
        mse = loss_function(reconstructed, image)
        total_mse += mse.item() * image.size(0)
        num_samples += image.size(0)

# вычисляем среднее значение MSE по всей тестовой выборке
average_mse = total_mse / num_samples
print(f"Average MSE on test dataset: {average_mse}")

# показываем некоторые результаты восстановления
# на тестовой выборке
for i,(image, _) in enumerate(testDataLoader):
    # Как и раньше, превращаем матрицу в вектор
    reshaped_image = image.reshape(-1, 28*28)

    # подаём вектор на автокодировщик
    reconstructed = model(reshaped_image)
    # получаем изображение из результата автокодировщика
    img = reconstructed.reshape(-1, 28, 28)

    # выводим исходное и итоговое изображения
    f, axarr = plt.subplots(2,1)
    raw_image = image[0].detach().numpy()[0,:,:]
    rec_image = img[0].detach().numpy()
    axarr[0].imshow(raw_image, cmap='gray')
    axarr[1].imshow(rec_image, cmap='gray')
    plt.show(block=True)
    if i == 3: # выводим 3+1 первых изображений, потом выходим.
        break