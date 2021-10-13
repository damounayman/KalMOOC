# Method of least squares
import numpy as np
import matplotlib.pyplot as plt

M = np.array([[9 ,  -3 ,    1],
              [1 ,  -1 ,    1],
              [0 ,   0,     0], 
              [4 ,   2,     1], 
              [9 ,   3,     1], 
              [36,   6,     1]])

y = np.array([[17], [3], [1], [5], [11], [46]])
t = [-3,-1,0,2,3,6]
estimator= np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)
p_hat = np.dot(estimator, y)

print("l'estimation des paramètres au sens des moindres-carrés:")
print("p1: ", p_hat[0],"\np2: ",p_hat[1],"\np3: ",p_hat[2])

# les mesures filtrées et vecteur résiduel
y_hat = np.dot(M, p_hat)
r = y_hat - y
print("\n")
print("Les mesures filtrées:\n", y_hat)
print("\n")
print("Vecteur résiduel:\n", r)

plt.figure("Gradient de la fonction f")
plt.plot(t, y_hat, label='y_hat')
plt.plot(t, y, label='y')
plt.xlabel('t')
plt.legend(loc='best') 
plt.show()