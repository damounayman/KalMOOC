import numpy as np

# Method of least squares
M = np.array([[4, 0],
             [10, 1],
             [10, 5],
             [13, 5],
             [15, 3]])

y = np.array([[5], [10], [8], [14], [17]])

estimator = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)
p_hat = np.dot(estimator, y)
print(" l'estimation des paramètres au sens des moindres-carrés")
print(p_hat)

# En déduire une estimation de la vitesse, U = 20V et Tr =10 Nm
U = 20
Tr = 10
print("En déduire une estimation de la vitesse, U = 20V et Tr =10 Nm")
Omega_estim = p_hat[0] * U + p_hat[1] * Tr
print(Omega_estim)