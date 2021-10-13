# Method of least squares
import numpy as np
import matplotlib.pyplot as plt

##############################
# Premier fonction f1(x)=x*y##
##############################

def f1(x,y):
    return x*y

def gradient(x,y):
    return y, x

Q = np.array([[0.0, 0.5],
              [0.5, 0.0]])
L = np.array([0, 0])
c = 0.0

x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
dx, dy = gradient(x, y)

# Gradients
plt.figure("Gradient de la fonction f")
plt.quiver(x, y, dx, dy)

# Courbe de niveaux
z = f1(x, y)
fig = plt.figure("Courbe de niveaux")
ax = fig.add_subplot(121, projection="3d", title="courbes de niveaux de f")
ax.contour(x, y, z, 10, cmap="jet")
ax2 = fig.add_subplot(122, projection="3d", title="f(x)")
ax2.plot_surface(x, y, z, cmap="jet")
plt.show()

##############################
# seconde fonction ###########
##############################
def f2(x, y):
    return 2*x**2 + x*y + 4 * y**2 + y - x + 3

Q = np.array([[2.0, 0.5],
              [0.5, 4.0]])
L = np.array([-1.0, 1.0])
c = 3.0

def gradient_f2(X):
    return 2 * X @ Q + L

X = np.dstack((x, y))
deriv_f2 = gradient_f2(X)

# gradients
plt.figure("Gradient de f2")
plt.quiver(x, y, deriv_f2[:, :, 0], deriv_f2[:, :, 1])

# courbe de niveaux
z = f2(x, y)
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d", title="courbes de niveaux de f2")
ax.contour(x, y, z, 10, cmap="jet")
ax2 = fig.add_subplot(122, projection="3d", title="f2(x)")
ax2.plot_surface(x, y, z, cmap="jet")
plt.show()

minimum = -L @ np.linalg.inv(Q) * 0.5
print("Minimum de la f2: ", minimum)