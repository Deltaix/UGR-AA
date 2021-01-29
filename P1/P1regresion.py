# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Juan Pablo García Sánchez
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, exp, symbols, sin

np.random.seed(1)

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Apartado 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    temp = np.dot(x, w) - y
    transpuesta = np.transpose(temp)
    matriz = np.dot(transpuesta, temp)
    return matriz/len(x)

# Gradiente Descendente Estocastico
def sgd(x, y):
    w = np.zeros(3)
    N = 32
    error = 1.0
    eta = 0.01
    mejor_peso = np.zeros(3)
    menor_error = 1.0
    iteraciones = 0
    while iteraciones < 100 and error > 0.05:
        batch = []
        valores = []
        for i in range(N):
            aux = np.random.choice(len(x))
            batch.append(x[aux])
            valores.append(y[aux])
        
        for i in range(len(batch)):
            prediccion = np.dot(batch, w)
            w = w - (1/len(batch)) * eta * (np.dot(np.transpose(batch), prediccion-valores))
        
        error = Err(batch, valores, w)
        
        if error < menor_error:
            mejor_peso = w
            menor_error = error
        
        iteraciones += 1
        
    return mejor_peso

# Pseudoinversa	
def pseudoinverse(x, y):
    transpuesta = np.transpose(x)
    inversa = np.linalg.inv(np.dot(transpuesta, x))
    pseudo_inversa = np.dot(inversa, transpuesta)
    return np.dot(pseudo_inversa, y)

def clasificar(x, w):
    y = []
    for i in x:
        aux = np.dot(i, w)
        if aux > 0:
            y.append(label5)
        else:
            y.append(label1)
    
    return y

def asignar_etiquetas(x1, x2, y1, y2, x, labels):
    for i in range(len(x_test)):
        if labels[i] == label1:
            x1.append(x[i][1])
            y1.append(x[i][2])
        else:
            x2.append(x[i][1])
            y2.append(x[i][2])


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


x1, x2, y1, y2 = [], [], [], []
asignar_etiquetas(x1, x2, y1, y2, x_test, y_test)

plt.figure(3)
plt.scatter(x1, y1, label='-1')
plt.scatter(x2, y2, label='1')
plt.title('Clasificacion real de los puntos')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

w = pseudoinverse(x, y)

print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

etiquetas = clasificar(x_test, w)
x1, x2, y1, y2 = [], [], [], []
asignar_etiquetas(x1, x2, y1, y2, x_test, etiquetas)

plt.figure(4)
plt.scatter(x1, y1, label='-1')
plt.scatter(x2, y2, label='1')
plt.title('Pseudoinversa')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

w = sgd(x, y)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

etiquetas = clasificar(x_test, w)
x1, x2, y1, y2 = [], [], [], []
asignar_etiquetas(x1, x2, y1, y2, x_test, etiquetas)

plt.figure(5)
plt.scatter(x1, y1, label='-1')
plt.scatter(x2, y2, label='1')
plt.title('Gradiente descendente estocástico')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Apartado 2.a\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

N = 1000
limite = 1
dim = 3
train = simula_unif(N, dim, limite)
x_train = []
y_train = []

for i in range(N):
    train[i][0] = 1.0
    
for i in train:
    x_train.append(i[1])
    y_train.append(i[2])

plt.figure(6)
plt.scatter(x_train, y_train)
plt.title('Puntos generados')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Apartado 2.b\n')

def funcion(x1, x2):
    return np.sign((x1 - 0.2)**2 + x2**2 - 0.6)

labels = []

for i in train:
    labels.append(funcion(i[1], i[2]))

for i in range(int(0.1 * N)):
    indice = np.random.choice(N)
    if labels[indice] == 1:
        labels[indice] = -1
    else:
        labels[indice] = 1

x1, x2, y1, y2 = [], [], [], []
asignar_etiquetas(x1, x2, y1, y2, train, labels)

plt.figure(7)
plt.scatter(x1, y1, label='negativo')
plt.scatter(x2, y2, label='positivo')
plt.title('Clasificacion con f(x1, x2)')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Apartado 2.c\n')

train_copia = np.copy(train)
w = sgd(train_copia, labels)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(train_copia, labels, w))

labels = clasificar(train_copia, w)

x1, x2, y1, y2 = [], [], [], []
asignar_etiquetas(x1, x2, y1, y2, train_copia, labels)

plt.figure(8)
plt.scatter(x1, y1, label='negativo')
plt.scatter(x2, y2, label='positivo')
plt.title('Gradiente descendente estocástico')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Apartado 2.d\n')

Ein, Eout = 0.0, 0.0

for iteracion in range(1000):
    train = simula_unif(N, dim, limite)
    test = simula_unif(N, dim, limite)
    
    for i in range(N):
        train[i][0] = 1.0
        test[i][0] = 1.0
        
    labels_train, labels_test = [], []
    
    for i in train:
        labels_train.append(funcion(i[1], i[2]))
        
    for i in test:
        labels_test.append(funcion(i[1], i[2]))
    
    for i in range(int(0.1 * N)):
        indice = np.random.choice(N)
        if labels_train[indice] == 1:
            labels_train[indice] = -1
        else:
            labels_train[indice] = 1
            
        if labels_test[indice] == 1:
            labels_test[indice] = -1
        else:
            labels_test[indice] = 1
            
    w = sgd(train, labels)
    
    Ein += Err(train, labels_train, w)
    Eout += Err(test, labels_test, w)
    if iteracion % 100 == 0:
    	print("Iteracion ", iteracion)
    
print("Ein medio: ", Ein/N)
print("Eout medio: ", Eout/N)