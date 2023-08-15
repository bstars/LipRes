import torch
from torch import nn
import numpy as np

from layer import SandwichFC, SandwichConv
from ori_layers import LipResFC, LipResConv

def test_LipResFC():
	print('------------------ test_LipResFC ------------------')
	nin = 6
	nout = 10
	for i in range(50):
		L = np.random.uniform(0, 0.5, 1)[0]
		liplin = LipResFC(nin, nout, L=L)
		x = torch.randn(2, nin)
		y = liplin(x)

		dx = np.sqrt(torch.sum((x[0, :] - x[1, :]) ** 2).item())
		dy = np.sqrt(torch.sum((y[0, :] - y[1, :]) ** 2).item())

		print(L, dy / dx, dx, dy)
		assert dy <= L * dx


def test_SandwichConv():
	print('------------------ test_LipResConv ------------------')
	cin = 3
	cout = 8
	ksize = 3
	for i in range(50):
		L = np.random.uniform(0, 0.5, 1)[0]
		lipconv = LipResConv(cin, cout, ksize, L=L, strided= i < 25)
		x = torch.randn(2, cin, 32, 32)
		y = lipconv(x)

		dx = np.sqrt(torch.sum((x[0, :] - x[1, :]) ** 2).item())
		dy = np.sqrt(torch.sum((y[0, :] - y[1, :]) ** 2).item())

		print(L, dy / dx, dx, dy, y.shape)
		assert dy <= L * dx

if __name__ == '__main__':
	test_LipResFC()
	pass