import numpy as np


U = np.array([2, -1, 1, -1, 1, -1, 1, -1])
Y = np.array([0, -1, -2, 3, 7, 11, 16, 36])
M = np.array([-Y[1:7], -Y[0:6], U[1:7], U[0:6]]).T

y = np.array([Y[2:8]]).reshape(6,1)

K = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)
p_hat = np.dot(K, y)

print(" L'estimation des paramètres au sens des moindres-carrés:\n", p_hat)
y_hat = np.dot(M, p_hat)
r = y_hat - y
print("Les mesures filtrées:\n", y_hat)

print("Vecteur résiduel:\n", r)