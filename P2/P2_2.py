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

def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# Algoritmo Perceptron
def ajusta_PLA(datos, label, max_iter, wini):
    w = wini
    contador = 0
    cambios = True

    while contador < max_iter and cambios:
        cambios = False
        
        for i in range(len(datos)):
            if np.sign(np.dot(np.transpose(w),datos[i])) != label[i]:
                w = w + label[i] * datos[i]
                cambios = True
                
        contador += 1

    return w, contador

print("Apartado 2.a.1")

# Crea 100 puntos aleatorios con distribución uniforme
puntos = simula_unif(100, 2, [-50, 50])

# Añade una columna de 1 que será el término independiente
puntos = np.transpose(puntos)
unos = np.ones(len(puntos[0]))
puntos = np.insert(puntos, 0, unos, axis=0)
puntos = np.transpose(puntos)

# Crea una recta que corta al cuadrado que los contiene
a, b = simula_recta([-50, 50])

# Toma las etiquetas de cada punto y los separa entre positivos o negativos
# según su distancia a la recta
etiquetas, positivos, negativos = [], [], []
for i in puntos:
    valor = f(i[1], i[2], a, b)
    etiquetas.append(valor)
    
    if valor == 1:
        positivos.append(i)
    else:
        negativos.append(i)

pt = np.transpose(positivos)
px, py = pt[1], pt[2]

nt = np.transpose(negativos)
nx, ny = nt[1], nt[2]

# Se inicializan los pesos a 0 y se ejecuta PLA
w = np.zeros(len(puntos[0]))
pesos, iteraciones = ajusta_PLA(puntos, etiquetas, 10000, w)

print("Iteraciones: ", iteraciones)

input("\n--- Pulsar tecla para continuar ---\n")

# Random initializations
iterations = []
for i in range(0, 10):
    w = np.random.rand(len(puntos[0]))
    pesos, iteraciones = ajusta_PLA(puntos, etiquetas, 10000, w)
    print("Iteraciones: ", iteraciones)
    iterations.append(iteraciones)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

# Se calcula la pendiente y el término independiente de la recta
# calculada por el algoritmo PLA
a = -(pesos[0]/pesos[2])/(pesos[0]/pesos[1])
b = -pesos[0]/pesos[2]

x = np.linspace(-50, 50, 100)
y = a*x + b

plt.figure(1)
plt.ylim(-50.0, 50.0)
plt.plot(x, y)
plt.scatter(px, py, label="positivos")
plt.scatter(nx, ny, label="negativos")
plt.title("Clasificacion según el signo la distancia a la recta y = " + str(a) + "x + " + str(b) + " con ruido")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
print("Apartado 2.a.2")

pruido, nruido = [], []

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

puntos = np.concatenate((positivos, negativos))

pt = np.transpose(positivos)
px, py = pt[1], pt[2]

nt = np.transpose(negativos)
nx, ny = nt[1], nt[2]

w = np.zeros(len(puntos[0]))
pesos, iteraciones = ajusta_PLA(puntos, etiquetas, 10000, w)

print("Iteraciones: ", iteraciones)

a = -(pesos[0]/pesos[2])/(pesos[0]/pesos[1])
b = -pesos[0]/pesos[2]

x = np.linspace(-50, 50, 100)
y = a*x + b

input("\n--- Pulsar tecla para continuar ---\n")

# Random initializations
iterations = []
for i in range(0,10):
    w = np.random.rand(len(puntos[0]))
    pesos, iteraciones = ajusta_PLA(puntos, etiquetas, 10000, w)
    print("Iteraciones: ", iteraciones)
    iterations.append(iteraciones)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

plt.figure(2)
plt.ylim(-50.0, 50.0)
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

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sigmoide(x):
    return 1/(np.exp(-x)+1)

# Probabilidad de que X pertenezca a una etiqueta u otra
def probabilidad(x, w):
    return sigmoide(np.dot(x, w))

def actualizar_pesos(X, y, w, eta):
    predicciones = probabilidad(X, w)
    gradiente = np.dot(np.transpose(X), predicciones - y)
    w -= gradiente * eta / len(X)
    return w

# Funcion para calcular el error
def Err(x, y):
    resta = np.subtract(x, y)
    return float(np.count_nonzero(resta)) / len(resta)

def sgdRL(X, y, w, eta):
    contador = 0
    pesos = w
    pesos_anterior = np.ones(3)
    resta = pesos_anterior - pesos
    
    X_aux, y_aux = np.copy(X), np.copy(y)
    
    # Itera mientras que la diferencia entre los pesos calculados y los anteriores
    # sea mayor a 0.01 o hasta que se hagan 50000 iteraciones
    while np.linalg.norm(resta) > 0.01 and contador < 50000:
        pesos_anterior = np.copy(pesos)
        
        # Con estado nos aseguramos de que se haga la misma permutación
        # para X_aux y y_aux
        estado = np.random.get_state()
        np.random.shuffle(X_aux)
        np.random.set_state(estado)
        np.random.shuffle(y_aux)
        
        # Minibatches de 16 elementos
        i = 16
        while (i < len(X)):
            batch_x, batch_y = X_aux[i-16:i], y_aux[i-16:i]
            pesos = actualizar_pesos(batch_x, batch_y, pesos, eta)
            i += 16
        
        resta = pesos_anterior - pesos
        contador += 1
    
    return pesos, contador

# Crea 100 puntos aleatorios con distribución uniforme
X = simula_unif(100, 2, [0,2])

# Añade una columna de 1 que será el término independiente
X = np.transpose(X)
unos = np.ones(len(X[0]))
X = np.insert(X, 0, unos, axis=0)
X = np.transpose(X)

# Genera 2 puntos y calcula la recta que pasa por ellos
punto1 = simula_unif(1, 2, [0,2])
punto2 = simula_unif(1, 2, [0,2])

a = (punto2[0][1] - punto1[0][1]) / (punto2[0][0] - punto1[0][0])
b = X[0][2] - a * X[0][1]

# Toma las etiquetas de cada punto según su distancia a la recta
y = []
for i in range(len(X)):
    valor = f(X[i][1], X[i][2], a, b)
    if valor == 1.0:
        y.append(valor)
    else:
        y.append(0.0) 

# Inicializamos el vector de pesos a 0 y el learning rate a 0.1
w = np.zeros(3)
eta = 0.1

# Calcula el vector de pesos con el algoritmo
pesos, iteraciones = sgdRL(X, y, w, eta)     

print(pesos)
print("Número de iteraciones necesarias: ", iteraciones, "\n")

# Separa las etiquetas positivas y negativas
positivos, negativos = [], []
for i in range(len(y)):
    if y[i] == 1.0:
        positivos.append(X[i])
    else:
        negativos.append(X[i]) 
    
pt = np.transpose(positivos)
px, py = pt[1], pt[2]

nt = np.transpose(negativos)
nx, ny = nt[1], nt[2]

a = -(pesos[0]/pesos[2])/(pesos[0]/pesos[1])
b = -pesos[0]/pesos[2]

x = np.linspace(0,2,100)
y = a*x + b

#Generación del gráfico
plt.figure(3)
plt.ylim(0.0, 2.0)
plt.plot(x,y)
plt.scatter(px, py, label="positivos")
plt.scatter(nx, ny, label="negativos")
plt.title("Frontera obtenida mediante SGD")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Apartado 2.b\n")

# Crea 100 puntos aleatorios con distribución uniforme para la prueba
test_x = simula_unif(1000, 2, [0,2])

# Añade una columna de 1 que será el término independiente
test_x = np.transpose(test_x)
unos = np.ones(len(test_x[0]))
test_x = np.insert(test_x, 0, unos, axis=0)
test_x = np.transpose(test_x)

# Toma las etiquetas de cada punto según su distancia a la recta
test_y = []
for i in range(len(test_x)):
    valor = f(test_x[i][1], test_x[i][2], a, b)
    if valor == 1.0:
        test_y.append(valor)
    else:
        test_y.append(0.0)

# Calcula las predicciones según los pesos calculados con sgdRL
predicciones = []
for i in range(len(test_x)):
    pred = np.dot(np.transpose(pesos), test_x[i])
    if pred >= 0.5:
        predicciones.append(1.0)
    else:
        predicciones.append(0.0)
    
# Cálcula el error de la prueba
Eout = Err(predicciones, test_y)
print("Eout: ", Eout)

positivos, negativos = [], []
for i in range(len(predicciones)):
    if predicciones[i] == 1.0:
        positivos.append(test_x[i])
    else:
        negativos.append(test_x[i]) 
    
pt = np.transpose(positivos)
px, py = pt[1], pt[2]

nt = np.transpose(negativos)
nx, ny = nt[1], nt[2]

plt.figure(4)
plt.ylim(0.0, 2.0)
plt.plot(x, y)
plt.scatter(px, py, label="positivos")
plt.scatter(nx, ny, label="negativos")
plt.title("Frontera obtenida mediante SGD")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")