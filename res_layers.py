import numpy as np
import torch
from torch import nn

class LipResLinear(nn.Module):

	@staticmethod
	def cayley(W):
		n, nout = W.shape
		WA, WB = W[:nout, :], W[nout:, :]
		I = torch.eye(nout).to(W.device)
		Z = WA - WA.T + WB.T @ WB
		iIpZ = torch.inverse(I + Z)
		GT = (I - Z) @ iIpZ
		HT = -2 * WB @ iIpZ
		return GT, HT

	def __init__(self, nin, nout, activation=nn.Tanh, L=1.):
		super().__init__()
		self.W = nn.Parameter(torch.randn(nin + nout, nout), requires_grad=True)
		self.lamb = nn.Parameter(torch.randn(nout), requires_grad=True)
		self.b = nn.Parameter(torch.zeros(nout), requires_grad=True)
		self.sigma = activation()
		self.L = L

	def forward(self, x):
		"""
		:param x: torch.Tensor, [batch_size, nin]
		:return:
		:rtype:
		"""
		x = x.T

		GT, HT = self.cayley(self.W)
		Lamb = 0.5 + torch.exp(self.lamb)
		W = -1 * self.L / Lamb[:,None] * (GT @ HT.T)

		y = self.L * HT.T @ x + GT.T @ self.sigma(W @ x + self.b[:,None])

		return y.T

class LipResConv(nn.Module):

	@staticmethod
	def cayley(W):
		"""
		:param W: [_, cout + cin, cout]
		:return:
		:rtype:
		"""
		_, c, cout = W.shape
		cin = c - cout
		U = W[:, :cout, :]
		V = W[:, cout:, :]
		I = torch.eye(cout, dtype=W.dtype, device=W.device)[None, :, :]
		A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
		iIpA = torch.inverse(I + A)
		GTs = iIpA @ (I - A)
		HTs = -2 * V @ iIpA
		return GTs, HTs

	@staticmethod
	def fft_shift_matrix(n, shift_amount):
		shift = torch.arange(0, n).repeat((n, 1)).to()
		shift = shift + shift.T
		return torch.exp(1j * 2 * np.pi * shift_amount * shift / n)

	def __init__(self, cin, cout, kernel_size, strided=True, activation=nn.Tanh, L=1.):
		super().__init__()
		assert kernel_size % 2 == 1, "kernel size must be odd"

		self.cin = cin
		self.cout = cout
		self.k = kernel_size
		self.shift_amount = (self.k - 1) // 2
		self.strided = strided

		self.W = nn.Parameter(torch.randn(cin + cout, cout, self.k, self.k), requires_grad=True)
		self.lamb = nn.Parameter(torch.randn(cout), requires_grad=True)
		self.b = nn.Parameter(torch.zeros(cout), requires_grad=True)
		self.sigma = activation()
		self.L = L

	def forward(self, x):
		"""
		:param x: torch.Tensor, [batch_size, cin, n, n]
		"""
		batch, c, n, _ = x.shape

		# Compute the cayley transform of diagonal D
		wfft =  self.fft_shift_matrix(n, -self.shift_amount).to(self.W.device) * torch.fft.fft2(self.W, (n, n)).conj() # [cout + cin, cout, n, n]
		wfft = wfft.reshape(self.cin + self.cout, self.cout, -1) # [cout + cin, cout, n^2]
		wfft = wfft.permute(2, 0, 1) # [n^2, cout + cin, cout]
		GTs, HTs = self.cayley(wfft) # [n^2, cout, cout], [n^2, cin, cout]
		GTHs = torch.bmm( GTs, HTs.transpose(1, 2).conj() ) # [n^2, cout, cout]

		# x
		xfft = torch.fft.fft2(x, (n, n)) # [batch, cin, n, n]
		xfft = xfft.reshape(batch, self.cin, -1) # [batch, cin, n^2]
		xfft = xfft.permute(2, 1, 0) # [n^2, cin, batch]

		# compute the affine function in \sigma
		Lamb = 0.5 + torch.exp(self.lamb)
		GTHx_fft = torch.bmm(GTHs, xfft) # [n^2, cout, batch]
		GTHx = GTHx_fft.permute(2, 1, 0) # [batch, cout, n^2]
		GTHx = GTHx.reshape(batch, self.cout, n, n) # [batch, cout, n, n]
		GTHx = torch.fft.ifft2(GTHx) # [batch, cout, n, n]

		assert GTHx.imag.abs().max() < 1e-5 # make sure the convolution output is real
		# print(GTHx.imag.abs().max())
		GTHx = GTHx.real
		affine = -1 * self.L / Lamb[None, :, None, None] * GTHx + self.b[None, :, None, None] # [batch, cout, n, n]

		# compute G \sigma( ... )
		sig = self.sigma(affine) # [batch, cout, n, n]
		sig_fft = torch.fft.fft2(sig, (n, n)) # [batch, cout, n, n]
		sig_fft = sig_fft.reshape(batch, self.cout, -1) # [batch, cout, n^2]
		sig_fft = sig_fft.permute(2, 1, 0) # [n^2, cout, batch]
		Gsig_fft = torch.bmm(GTs.transpose(1,2).conj(), sig_fft) # [n^2, cout, batch]
		Gsig = Gsig_fft.permute(2, 1, 0) # [batch, cout, n^2]
		Gsig = Gsig.reshape(batch, self.cout, n, n) # [batch, cout, n, n]
		Gsig = torch.fft.ifft2(Gsig) # [batch, cout, n, n]

		assert Gsig.imag.abs().max() < 1e-5 # make sure the convolution output is real
		# print(Gsig.imag.abs().max())
		Gsig = Gsig.real


		# compute the residual connection Hx
		Hx_fft = torch.bmm(HTs.transpose(1,2).conj(), xfft) # [n^2, cout, batch]
		Hx = Hx_fft.permute(2, 1, 0) # [batch, cout, n^2]
		Hx = Hx.reshape(batch, self.cout, n, n) # [batch, cout, n, n]
		Hx = torch.fft.ifft2(Hx) # [batch, cout, n, n]

		assert Hx.imag.abs().max() < 1e-5 # make sure the convolution output is real
		# print(Hx.imag.abs().max())
		Hx = Hx.real

		y = self.L * Hx + Gsig # [batch, cout, n, n]

		if self.strided:
			y = y[:, :, ::2, ::2]
		return y


if __name__ == '__main__':
# Test
	x = torch.randn(4, 3, 64, 64)
	conv = LipResConv(3, 8, 3)
	y = conv(x)




