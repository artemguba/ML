import math
import random

import torch
from torch import nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed(17)
torch.manual_seed(19)

# Параметры эллиптического параболоида
u = 1.5
v = 2.0

# создаём обучающую выборку
N = 512  # размер выборки
train_dataset = torch.zeros((N, 3))
curr_len = 0
while curr_len < N:
    # генерируем точки, равномерно распределённые в единичном круге
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = u * x**2 + v * y**2
    train_dataset[curr_len, 0] = x
    train_dataset[curr_len, 1] = y
    train_dataset[curr_len, 2] = z
    curr_len += 1

# делаем номер класса для примеров из обучающей выборки
# единичным. Сгенерированные примеры будут иметь этот номер равным 0
train_labels = torch.ones(N)

train_set = [(train_dataset[i], train_labels[i]) for i in range(N)]

# готовим загрузчик данных
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

# параметры обучения: скорость, число эпох, функция потерь
lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

# Эксперимент 1: Влияние числа скрытых параметров
hidden_param_nums = [2, 4, 8, 16]

for hidden_param_num in hidden_param_nums:
    print(f"Experiment 1: Training with hidden_param_num = {hidden_param_num}")

    class Generator(nn.Module):
        def __init__(self, hidden_param_num):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(hidden_param_num, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2))

        def forward(self, x):
            output = self.model(x)
            x = output[:, 0]
            y = output[:, 1]
            z = u * x**2 + v * y**2
            return torch.stack((x, y, z), dim=1)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid())

        def forward(self, x):
            output = self.model(x)
            return output

    generator = Generator(hidden_param_num)
    discriminator = Discriminator()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    losses_d = []
    losses_g = []

    for epoch in range(num_epochs):
        for n, (real_samples, real_samples_1d) in enumerate(train_loader):
            real_samples_labels = real_samples_1d.view(batch_size, 1)
            latent_space_samples = torch.randn((batch_size, hidden_param_num))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size,1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels,
                                            generated_samples_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator,
                                               all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            latent_space_samples = torch.randn((batch_size, hidden_param_num))

            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated,
                                           real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

        losses_d.append(loss_discriminator.item())
        losses_g.append(loss_generator.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    latent_space_samples = torch.randn((100, hidden_param_num))
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(train_dataset[:, 0], train_dataset[:, 1], train_dataset[:, 2], 'b')
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], 'r')
    plt.title(f"Generated points with hidden_param_num = {hidden_param_num}")
    plt.show()

    plt.figure()
    plt.plot(losses_d, label='Discriminator Loss')
    plt.plot(losses_g, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses with hidden_param_num = {hidden_param_num}')
    plt.legend()
    plt.show()

# Эксперимент 2: Влияние структуры сетей
print("Experiment 2: Training with modified network structure")

class GeneratorModified(nn.Module):
    def __init__(self, hidden_param_num):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_param_num, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, x):
        output = self.model(x)
        x = output[:, 0]
        y = output[:, 1]
        z = u * x**2 + v * y**2
        return torch.stack((x, y, z), dim=1)

class DiscriminatorModified(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        output = self.model(x)
        return output

generator = GeneratorModified(hidden_param_num)
discriminator = DiscriminatorModified()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

losses_d = []
losses_g = []

for epoch in range(num_epochs):
    for n, (real_samples, real_samples_1d) in enumerate(train_loader):
        real_samples_labels = real_samples_1d.view(batch_size, 1)
        latent_space_samples = torch.randn((batch_size, hidden_param_num))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size,1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels,
                                        generated_samples_labels))

        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator,
                                           all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        latent_space_samples = torch.randn((batch_size, hidden_param_num))

        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated,
                                       real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

    losses_d.append(loss_discriminator.item())
    losses_g.append(loss_generator.item())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator}")

latent_space_samples = torch.randn((100, hidden_param_num))
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(train_dataset[:, 0], train_dataset[:, 1], train_dataset[:, 2], 'b')
ax.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], 'r')
plt.title("Generated points with modified network structure")
plt.show()

plt.figure()
plt.plot(losses_d, label='Discriminator Loss')
plt.plot(losses_g, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses with modified network structure')
plt.legend()
plt.show()

# Эксперимент 3: Влияние Dropout
dropout_rates = [0.0, 0.3, 0.5, 0.7]

for dropout_rate in dropout_rates:
    print(f"Experiment 3: Training with dropout_rate = {dropout_rate}")

    class DiscriminatorDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1),
                nn.Sigmoid())

        def forward(self, x):
            output = self.model(x)
            return output

    generator = Generator(hidden_param_num)
    discriminator = DiscriminatorDropout()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    losses_d = []
    losses_g = []

    for epoch in range(num_epochs):
        for n, (real_samples, real_samples_1d) in enumerate(train_loader):
            real_samples_labels = real_samples_1d.view(batch_size, 1)
            latent_space_samples = torch.randn((batch_size, hidden_param_num))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size,1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels,
                                            generated_samples_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator,
                                               all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            latent_space_samples = torch.randn((batch_size, hidden_param_num))

            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated,
                                           real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

        losses_d.append(loss_discriminator.item())
        losses_g.append(loss_generator.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    latent_space_samples = torch.randn((100, hidden_param_num))
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(train_dataset[:, 0], train_dataset[:, 1], train_dataset[:, 2], 'b')
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], 'r')
    plt.title(f"Generated points with dropout_rate = {dropout_rate}")
    plt.show()

    plt.figure()
    plt.plot(losses_d, label='Discriminator Loss')
    plt.plot(losses_g, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses with dropout_rate = {dropout_rate}')
    plt.legend()
    plt.show()
