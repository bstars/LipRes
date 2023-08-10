import torch
from torch import nn
import numpy as np

from res_layers import LipResLinear, LipResConv

# from layers import cayley_linear, LipLinear
#
#
# def test_lip_linear():
# 	nin = 6
# 	nout = 64
# 	for i in range(50):
# 		L = np.random.uniform(0, 1, 1)[0]
# 		liplin = LipLinear(nin, nout, activation=nn.Sigmoid, L=L)
# 		x = torch.randn(2, nin)
# 		y = liplin(x)
#
# 		dx2 = torch.sum( (x[0,:] - x[1,:]) ** 2, dim=0 ).item()
# 		dy2 = torch.sum( (y[0,:] - y[1,:]) ** 2, dim=0 ).item()
#
# 		print(L**2 * dx2, dy2)
# 		assert dy2 <= (L**2) * dx2
#
#
#
# test_lip_linear()


def test_LipResLinear():
	nin = 6
	nout = 10
	for i in range(50):
		L = np.random.uniform(0, 0.5, 1)[0]
		liplin = LipResLinear(nin, nout, activation=nn.Sigmoid, L=L)
		x = torch.randn(2, nin)
		y = liplin(x)

		dx = np.sqrt(torch.sum((x[0, :] - x[1, :]) ** 2).item())
		dy = np.sqrt(torch.sum((y[0, :] - y[1, :]) ** 2).item())

		print(L, dy / dx, dx, dy)
		assert dy <= L * dx

def test_LipResConv():
	cin = 3
	cout = 8
	ksize = 3
	for i in range(50):
		L = np.random.uniform(0, 0.5, 1)[0]
		lipconv = LipResConv(cin, cout, ksize, strided=False, activation=nn.Sigmoid, L=L)
		x = torch.randn(2, cin, 32, 32)
		y = lipconv(x)

		dx = np.sqrt(torch.sum((x[0, :] - x[1, :]) ** 2).item())
		dy = np.sqrt(torch.sum((y[0, :] - y[1, :]) ** 2).item())

		print(L, dy / dx, dx, dy, y.shape)
		assert dy <= L * dx

if __name__ == '__main__':
	# test_LipResLinear()
	test_LipResConv()
