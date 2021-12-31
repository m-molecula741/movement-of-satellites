import numpy as np
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import time
from joblib import Parallel, delayed
import multiprocessing


# класс модели
class TDynamicModel:
    # вектор в котором хранятся координаты и скорости по координатам модели
    TVector = np.zeros((6))

    # задание начальных значений при инициализации модели
    def __init__(self, x, y, z, Vx, Vy, Vz):
        self.TVector = x, y, z, Vx, Vy, Vz


# абстрактный класс интегрирования
class TAbstractIntegrator(ABC):
    # задаются начальное и конечное времени интегрирования
    # шаг интегрирования и для какой модели идет интегрирование
    def __init__(self, t0, tk, h, TDynamicModel):
        self.t0, self.tk, self.h = t0, tk, h
        self.TDynamicModel = TDynamicModel

    # вычисление правой части
    # deltaX, deltaY необходимы для вычисления правой части при небольших смещениях
    def SetRightParts(self, deltaX):
        RVector = np.zeros((6))
        TVector = np.zeros((6))
        if type(deltaX) == int:
            for i in range(6):
                TVector[i] = self.TDynamicModel.TVector[i] + deltaX
        else:
            TVector[:] = self.TDynamicModel.TVector[:] + deltaX[:]
        # вычисления правой части для координаит
        RVector[:3] = TVector[3:6]
        # вычисление правой части для скоростей по координатам
        G = 3.98603 * 10 ** 14
        r = (TVector[0] ** 2 + TVector[1] ** 2 + TVector[2] ** 2) ** 0.5
        RVector[3:6] = -G * TVector[0:3] / r ** 3
        return RVector

    # абстрактный метод для вычисления новых параметров модели с одним шагом интегрирования
    @abstractmethod
    def OneStep(self):
        pass

    # метод для вычисления интегрирования всех шагов
    def MoveTo(self):
        # создание массива у которого размер - количество шагов интегрирования
        # при начальном, конечном временах и шаге интегрирования + 1 для начальных значений
        move = np.zeros((int((self.tk - self.t0) / self.h + 1), 6))
        for i in range(int((self.tk - self.t0) / self.h + 1)):
            # запись текущих параметров модели
            move[i, :] = self.TDynamicModel.TVector[:]
            # вычисление параметров с одним шагом интегрирования
            self.OneStep()
        return move


# класс интегрирования Эйлера
class TEuler(TAbstractIntegrator):
    def OneStep(self):
        # прибавляем к текущим параметрам модели шаг интегрирования Эйлера
        self.TDynamicModel.TVector += self.h * self.SetRightParts(0)


# класс интегрирования Рунге-Кутта
class TRungeKutta(TAbstractIntegrator):
    def OneStep(self):
        k = np.zeros((4, 6))
        # вычисление коэф-тов Рунге-Кутта
        k[0, :] = self.SetRightParts(0)
        k[1, :] = self.SetRightParts(k[0, :] / 2)
        k[2, :] = self.SetRightParts(k[1, :] / 2)
        k[3, :] = self.SetRightParts(k[2, :])
        # Прибавляем к текущим параметрам модели шаг интегирования Рунге-Кутты
        self.TDynamicModel.TVector += self.h * (k[0, :] + 2 * k[1, :] + 2 * k[2, :] + k[3, :]) / 6.


# создание спутников с различными начальными параметрами
sputnik1 = TDynamicModel(42164000, 0, 0, 0, 3066, 0)
sputnik2 = TDynamicModel(42164000, 0, 0, 0, 3066, 0)

# задание начального, конечного времен и шага интегрирования
t0 = 0
tk = 21600
h = 0.01

# создание интегрирования Эйлера для первого спутника
euler1 = TEuler(t0, tk, h, sputnik1)

# вычисление перемещения первого спутника
# с помощью интегрирования Эйлера
move1 = euler1.MoveTo()

# создание интегрирования Рунге-Кутта для первого спутника
RungeKutta1 = TRungeKutta(t0, tk, h, sputnik2)

# вычисление перемещения первого спутника
# с помощью интегрирования Рунге-Кутта
move2 = RungeKutta1.MoveTo()

# создание массива с временем движения спутников
t = np.zeros((int((tk - t0) / h + 1)))
for i in range(int((tk - t0) / h + 1)):
    t[i] = t0 + i * h
# задаем размер отображения графика
plt.figure(figsize=(15, 5))
# задаем по точкам графики перемещения спутников по оси х относительно времени
plt.plot(t, move1[:, 0])
plt.plot(t, move2[:, 0])
plt.title("график спутника (без замера времени) отн х")
plt.ylabel('x')
plt.xlabel('t')

# создание массива с временем движения спутников
t = np.zeros((int((tk - t0) / h + 1)))
for i in range(int((tk - t0) / h + 1)):
    t[i] = t0 + i * h
# задаем размер графика
plt.figure(figsize=(15, 5))
# задаем по точкам графики перемещения спутников по оси у относительно времени
plt.plot(t, move1[:, 1])
plt.plot(t, move2[:, 1])
plt.title("график спутника (без замера времени) отн у")
plt.ylabel('y')
plt.xlabel('t')

# задаем массив с временем движения спутников
t = np.zeros((int((tk - t0) / h + 1)))
for i in range(int((tk - t0) / h + 1)):
    t[i] = t0 + i * h
plt.figure(figsize=(15, 5))
# задаем по точкам графиики перемещения по оси z относительно времени
plt.plot(t, move1[:, 2])
plt.plot(t, move2[:, 2])
plt.title("график спутника (без замера времени) отн z")
plt.ylabel('z')
plt.xlabel('t')

# задаем размер отображения графика
fig = plt.figure(figsize=(15, 15))
# помогает создавать трехмерные графики
ax = fig.add_subplot(projection='3d')
# задаем по точкам графики перемещения спутников по осям х, у , z
surf = ax.plot(move1[:, 0], move1[:, 1], move1[:, 2])
surf = ax.plot(move2[:, 0], move2[:, 1], move2[:, 2])
# задание названий осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# задание начального, конечного и времен интегрирования
t0 = 0
tk = 122400
h = 0.1

# создание спутника
sputnik3 = TDynamicModel(42164000, 0, 0, 0, 3066, 0)
# создание для него свое интегрирование
RungeKutta3 = TRungeKutta(t0, tk, h, sputnik3)
# время начала работы вычисления интегрирования движения спутника
start_time = time.time()
# вычисление перемещения третьего спутника
# с помощью интегрирования Рунге-Кутты
more3 = RungeKutta3.MoveTo()
# время конца работы вычисления интегрирования движения спутника
end_time = time.time()
# время работы вычисления движения спутника
t1 = end_time - start_time
print('время работы спутника при одном потоке',t1)

# задание начального, конечного времен и шага интегрирования
t0 = 0
tk = 122400
h = 0.1

# создание спутников с различными начальными параметрами
sputnik2 = TDynamicModel(42164000, 0, 0, 0, 3066, 0)
sputnik3 = TDynamicModel(42164000, 0, 0, 0, 0, 3066)
sputnik4 = TDynamicModel(0, 42164000, 0, 0, 0, 3066)
sputnik5 = TDynamicModel(29814450, 29814450, 0, 0, 0, 3066)
sputnik6 = TDynamicModel(29814450, -29814450, 0, 0, 0, 3066)

# создание для каждого спутника свое интегрирование
RungeKutta2 = TRungeKutta(t0, tk, h, sputnik2)
RungeKutta3 = TRungeKutta(t0, tk, h, sputnik3)
RungeKutta4 = TRungeKutta(t0, tk, h, sputnik4)
RungeKutta5 = TRungeKutta(t0, tk, h, sputnik5)
RungeKutta6 = TRungeKutta(t0, tk, h, sputnik6)


# распараллеливание вычислений интегрирования движений спутников
def processInput(i):
    if i == 0:
        return RungeKutta2.MoveTo()
    elif i == 1:
        return RungeKutta3.MoveTo()
    elif i == 2:
        return RungeKutta4.MoveTo()
    elif i == 3:
        return RungeKutta5.MoveTo()
    else:
        return RungeKutta6.MoveTo()


inputs = range(5)

# время начала работы вычисления интегрирования  движений спутников
start_time = time.time()

# запись кол-ва ядер на ЭВМ
num_cores = multiprocessing.cpu_count()

# запись результатов распараллеливания
result = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

# время конца работы вычисления движения интегрироания спутников
end_time = time.time()
# время работы вычисления движений спутников при распараллеливании
t2 = end_time - start_time
print('время работы спутников при многопоточке',t2)

# создание массива с временем движения спутников
t = np.zeros((int((tk - t0) / h + 1)))
for i in range(int((tk - t0) / h + 1)):
    t[i] = t0 + i * h

print(result[3][:, 0])
# задаем размер отображения графика
plt.figure(figsize=(15, 5))

# задаем по точкам графики перемещения спутников по оси х относительно времени
for i in range(5):
    plt.plot(t, result[i][:, 0])


plt.title("график для спутников (с распарал) отн х")
plt.ylabel('x')
plt.xlabel('t')

# создание массива с временем двиижения спутников
t = np.zeros((int((tk - t0) / h + 1)))
for i in range(int((tk - t0) / h + 1)):
    t[i] = t0 + i * h

# задаем размер отображения графика
plt.figure(figsize=(15, 5))

# задаем по точкам графики перемещения спутников по оси у относительно времени
for i in range(5):
    plt.plot(t, result[i][:, 1])




plt.title("график для спутников (с распарал) отн у")
plt.ylabel('y')
plt.xlabel('t')

# создаем массив с временем движения спутников
t = np.zeros((int((tk - t0) / h + 1)))
for i in range(int((tk - t0) / h + 1)):
    t[i] = t0 + i * h

# задаем размер отображаемого графика
plt.figure(figsize=(15, 5))

# задаем по точкам графики перемещения спутников по оси z относительно времени
for i in range(5):
    plt.plot(t, result[i][:, 2])




plt.title("график для спутников (с распарал) отн z")
plt.ylabel('z')
plt.xlabel('t')

# задаем размер отображаемого графика
fig = plt.figure(figsize=(15, 15))
# помогает создавать 3мерные графики
ax = fig.add_subplot(projection='3d')

# задаем по точкам графики перемещения спутников по осям х,у,z
for i in range(5):
    surf = ax.plot(result[i][:, 0], result[i][:, 1], result[i][:, 2])

# задаем название осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.show()

# сохранение изменений параметров второго спутника в формате csv
np.savetxt("sputnik2_RungeKutta.csv", result[1], delimiter=",")