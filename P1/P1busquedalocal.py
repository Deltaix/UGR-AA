# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Juan Pablo García Sánchez
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, exp, symbols, sin

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# Expresion de E(u,v) con simbolos
def E():
    x, y = symbols('x y')
    expresion = (x*exp(y) - 2*y*exp(-x))**2
    return expresion 

# Evaluacion de E(u, v) con valores numericos
def E2(u,v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    Eu = E()
    derivada = diff(Eu, 'x')
    return derivada.subs([('x',u), ('y',v)])
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    Ev = E()
    derivada = diff(Ev, 'y')
    return derivada.subs([('x',u), ('y',v)])

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#Gradiente Descendente
def gradient_descent(initial_point, maxIter, error2get, eta):
    iteraciones = 0
    err = 1.0
    punto = initial_point
    while iteraciones < maxIter and err > error2get:
        aux = punto
        punto = punto - eta * gradE(punto[0], punto[1]) #cálculo nuevo punto
        iteraciones += 1 
        err = abs(punto[1] - aux[1])
        
    return punto, iteraciones

eta = 0.1 #learning rate
maxIter = 10000000000
error2get = 1e-14 
initial_point = np.array([1.0, 1.0])
w, it = gradient_descent(initial_point, maxIter, error2get, eta)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
Z = E2(X,Y) #E_w([X, Y])
fig = plt.figure(1)
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet', zorder=0)
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
temp = E().subs([('x', min_point_[0]), ('y', min_point_[1])])
ax.plot(min_point_[0], min_point_[1], temp, 'r*', markersize=10, zorder=10)
ax.set(title='Ejercicio 1.2. Gradiente descendiente de E(u,v) con eta = 0.1')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("\n--- Pulse una tecla para pasar al siguiente apartado ---\n")
print("Apartado 3.a\n")

### APARTADO 3
# Expresion con simbolos
def f():
    x, y = symbols('x y')
    expresion = (x-2)**2 + 2*(y**2 + 2)**2 + 2*sin(2 * np.pi * x) * sin(2 * np.pi * y)
    return expresion

# Evaluacion numerica de la expresion
def f2(x,y):
    return (x-2)**2 + 2*(y**2 + 2)**2 + 2*sin(2 * np.pi * x) * sin(2 * np.pi * y)

# Derivada parcial de f(x,y) con respecto a x
def dFx(x,y):
    Fx = f()
    derivada = diff(Fx, 'x')
    return derivada.subs([('x', x), ('y', y)])
    
# Derivada parcial de f(x,y) con respecto a y
def dFy(x,y):
    Fy = f()
    derivada = diff(Fy, 'y')
    return derivada.subs([('x', x), ('y', y)])

# Gradiente de f(x,y)
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])

def gradient_descent_f(maxIter, eta, error2get, puntos):
    iteraciones = 0
    err = 1.0
    punto = puntos[-1]
    while iteraciones < maxIter and err > error2get:
        aux = punto
        punto = punto - eta * gradF(punto[0], punto[1]) #cálculo nuevo punto
        iteraciones += 1 
        err = abs(punto[1] - aux[1])
        puntos.append(punto)
        
    return punto, iteraciones

## APARTADO 1.3.a
# Con eta = 0.01
initial_point = [1, -1]
eta = 0.01
maxIter = 50
puntos = [[]]
error2get = 1e-14 
puntos[0] = initial_point
w, it = gradient_descent_f(maxIter, eta, error2get, puntos)

print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

puntos.pop()
valores = []
for i in puntos:
    valores.append(f2(i[0], i[1]))

plt.figure(2)
plt.plot(range(it), valores)
plt.title('Ejercicio 1.3. Gradiente descendiente de f(x,y) con eta = 0.01')
plt.xlabel('Numero de iteraciones')
plt.ylabel('Valor de la funcion')
plt.show()

input("\n--- Pulse una tecla para pasar al siguiente apartado ---\n")

'''
# Con eta = 0.1
initial_point = [1, -1]
eta = 0.1
maxIter = 50
puntos = [[]]
puntos[0] = initial_point
w, it = gradient_descent_f(maxIter, eta, error2get, puntos)

print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

puntos.pop()
valores = []
for i in puntos:
    valores.append(f2(i[0], i[1]))

plt.figure(3)
plt.plot(range(maxIter), valores)
plt.title('Ejercicio 1.3. Gradiente descendiente de f(x,y) con eta = 0.01')
plt.xlabel('Numero de iteraciones')
plt.ylabel('Valor de la funcion')
plt.show()

input("\n--- Pulse una tecla para pasar al siguiente apartado ---\n")
'''

print("\nApartado 3.b")
## APARTADO 1.3.b
# Punto (2.1, -2.1)
eta = 0.01
initial_point = [2.1, -2.1]
puntos = [[]]
puntos[0] = initial_point
a, ita = gradient_descent_f(maxIter, eta, error2get, puntos)
valor1 = f2(a[0], a[1])

print ('\nPunto inicial: ', initial_point)
print ('Numero de iteraciones: ', ita)
print ('Coordenadas: (', a[0], ', ', a[1], ')')
print ('Valor de la funcion: ', valor1)

# Punto (3, -3)
initial_point = [3, -3]
puntos = [[]]
puntos[0] = initial_point
b, itb = gradient_descent_f(maxIter, eta, error2get, puntos)
valor2 = f2(b[0], b[1])

print ('\nPunto inicial: ', initial_point)
print ('Numero de iteraciones: ', itb)
print ('Coordenadas: (', b[0], ', ', b[1], ')')
print ('Valor de la funcion: ', valor2)

# Punto (1.5, 1.5)
initial_point = [1.5, 1.5]
puntos = [[]]
puntos[0] = initial_point
c, itc = gradient_descent_f(maxIter, eta, error2get, puntos)
valor3 = f2(c[0], c[1])

print ('\nPunto inicial: ', initial_point)
print ('Numero de iteraciones: ', itc)
print ('Coordenadas: (', c[0], ', ', c[1], ')')
print ('Valor de la funcion: ', valor3)

# Punto (1, -1)
initial_point = [1, -1]
puntos = [[]]
puntos[0] = initial_point
d, itd = gradient_descent_f(maxIter, eta, error2get, puntos)
valor4 = f2(d[0], d[1])

print ('\nPunto inicial: ', initial_point)
print ('Numero de iteraciones: ', itd)
print ('Coordenadas: (', d[0], ', ', d[1], ')')
print ('Valor de la funcion: ', valor4)

print ('\n(2.1, -2.1)\n(', a[0], ', ', a[1], ')\tf(x,y) = ', valor1, ' Iteraciones: ', ita)
print ('(3, -3)  \n(', b[0], ', ', b[1], ')\tf(x,y) = ', valor2, ' Iteraciones: ', itb)
print ('(1.5, 1.5)\n(', c[0], ', ', c[1], ')\tf(x,y) = ', valor3, ' Iteraciones: ', itc)
print ('(1, -1)  \n(', d[0], ', ', d[1], ')\tf(x,y) = ', valor4, ' Iteraciones: ', itd)