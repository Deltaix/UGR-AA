# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Juan Pablo García Sánchez
"""
import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

print("Apartado 1.1.a")

# Crea 50 puntos aleatorios distribuidos uniformemente
x = simula_unif(50, 2, [-50,50])
xt = np.transpose(x)
datosx, datosy = xt[0], xt[1]

# Muestra la nube de puntos
plt.figure(1)
plt.scatter(datosx, datosy)
plt.title("Nube de puntos con N=50, dim=2 y rango=[-50, 50] con simula_unif")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Apartado 1.1.b")

# Crea 50 puntos aleatorios siguiendo una distribución gaussiana
x = simula_gaus(50, 2, [5,7])
xt = np.transpose(x)
datosx, datosy = xt[0], xt[1]

# Muestra la nube de puntos
plt.figure(2)
plt.scatter(datosx, datosy)
plt.title("Nube de puntos con N=50, dim=2 y sigma=[5, 7] con simula_gaus")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

print("Apartado 1.2.a")

# Crea 100 puntos aleatorios distribuidos uniformemente
# y una recta que corte al cuadrado que los contiene
puntos = simula_unif(100, 2, [-50, 50])
a, b = simula_recta([-50, 50])

# Toma las etiquetas de cada punto y los separa entre positivos o negativos
# según su distancia a la recta
etiquetas, positivos, negativos = [], [], []
for i in puntos:
    valor = f(i[0], i[1], a, b)
    etiquetas.append(valor)
    
    if valor == 1:
        positivos.append(i)
    else:
        negativos.append(i)

pt = np.transpose(positivos)
px, py = pt[0], pt[1]

nt = np.transpose(negativos)
nx, ny = nt[0], nt[1]

# Crea 100 números equidistantes que serán la "x" de los puntos de la recta
# y calcula la coordenada "y" de cada uno de ellos
x = np.linspace(-50, 50, 100)
y = a*x + b

# Muestra la recta y la clasificación de los puntos generados según su etiqueta
plt.figure(3)
plt.plot(x, y)
plt.scatter(px, py, label="positivos")
plt.scatter(nx, ny, label="negativos")
plt.title("Clasificacion según el signo la distancia a la recta y = " + str(a) + "x + " + str(b))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

print("Apartado 1.2.b")

pruido, nruido = [], []

# Para añadir ruido, toma un 10% de las muestras positivas y las añade a las negativas
# y viceversa. El primer "if" comprueba que haya suficientes muestras
if len(positivos) * 0.1 >= 1:
    for i in range(int(len(positivos) * 0.1)):
        index = np.random.choice(len(positivos))
        nruido.append(positivos[index])
        positivos = np.delete(positivos, index, 0)
    negativos = np.concatenate((negativos, np.asarray(nruido)))

if len(negativos) * 0.1 >= 1:
    for i in range(int(len(negativos) * 0.1)):
        index = np.random.choice(len(negativos))
        pruido.append(negativos[index])
        negativos = np.delete(negativos, index, 0)
    positivos = np.concatenate((positivos, np.asarray(pruido)))

pt = np.transpose(positivos)
px, py = pt[0], pt[1]

nt = np.transpose(negativos)
nx, ny = nt[0], nt[1]

# Muestra la recta y la clasificación de los puntos con ruido
plt.figure(4)
plt.plot(x, y)
plt.scatter(px, py, label="positivos")
plt.scatter(nx, ny, label="negativos")
plt.title("Clasificacion según el signo la distancia a la recta y = " + str(a) + "x + " + str(b) + " con ruido")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2.c: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def f1(x, y):
    return (x - 10)**2 + (y - 20)**2 - 400

def f2(x, y):
    return 0.5*(x + 10)**2 + (y - 20)**2 - 400

def f3(x, y):
    return 0.5*(x - 10)**2 - (y + 20)**2 - 400

def f4(x, y):
    return y - 20*x**2 - 5*x + 3

print("Apartado 1.1.c")

frontera1, frontera2, frontera3, frontera4 = [], [], [], []

# Creamos una matriz 100x100 de números equidistantes para calcular las fronteras que dibuja cada función
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
x, y = np.meshgrid(x, y)
frontera1 = f1(x, y)
frontera2 = f2(x, y)
frontera3 = f3(x, y)
frontera4 = f4(x, y)

# Muestra los puntos con la frontera1
plt.figure(5)
plt.contour(x, y, frontera1, [0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por f1")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Muestra los puntos con la frontera2
plt.figure(6)
plt.contour(x, y, frontera2, [0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por f2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Muestra los puntos con la frontera3
plt.figure(7)
plt.contour(x, y, frontera3, [0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por f3")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Muestra los puntos con la frontera4
plt.figure(8)
plt.contour(x, y, frontera4, [0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por f4")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")