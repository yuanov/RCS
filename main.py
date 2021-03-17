import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from math import sqrt
from tqdm import tqdm
from cmath import pi, exp
from scipy.integrate import dblquad

lambd = 1
k = 2 * pi / lambd
z = lambd * 5
E0 = 1
u = E0 * exp(-1j * k * z)

ksi_brd = [-1.5, 1.5]
eta_brd = [-1.5, 1.5]


def field(x, y):
    def func_real(ksi, eta):
        r = sqrt((ksi - x) ** 2 + (eta - y) ** 2 + z ** 2)
        expr1 = exp(-1j * k * r) / r * (-1j * k * u)
        expr2 = u * (-(1j * k / r + 1 / r ** 2) * exp(-1j * k * r) * z / r)
        return 1 / (4 * pi) * (expr1 - expr2).real

    def func_imag(ksi, eta):
        r = sqrt((ksi - x) ** 2 + (eta - y) ** 2 + z ** 2)
        expr1 = exp(-1j * k * r) / r * (-1j * k * u)
        expr2 = u * (-(1j * k / r + 1 / r ** 2) * exp(-1j * k * r) * z / r)
        return 1 / (4 * pi) * (expr1 - expr2).imag
    # r = 1.5
    # return dblquad(func_real, -r, r, lambda eta: -sqrt(r**2 - eta ** 2), lambda eta: sqrt(r**2 - eta ** 2)), \
    #        dblquad(func_imag, -r, r, lambda eta: -sqrt(r**2 - eta ** 2), lambda eta: sqrt(r**2 - eta ** 2))

    return dblquad(func_real, eta_brd[0], eta_brd[1], lambda eta: ksi_brd[0], lambda eta: ksi_brd[1]), \
           dblquad(func_imag, eta_brd[0], eta_brd[1], lambda eta: ksi_brd[0], lambda eta: ksi_brd[1])


num_x = 20
num_y = 20
xs = linspace(-20, 20, num_x)
ys = linspace(-20, 20, num_y)
zs = np.zeros((num_y, num_x))
for i, x in tqdm(enumerate(xs), total=num_x):
    for j, y in enumerate(ys):
        res = field(x, y)
        zs[j][i] = abs(res[0][0] + 1j * res[1][0])

plt.figure(figsize=(9.6, 5))
font = {'size': 14}
plt.rc('font', **font)
cc = plt.contourf(xs, ys, np.abs(zs), cmap="coolwarm", levels=100)
plt.colorbar(cc)
plt.title('E', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.show()
