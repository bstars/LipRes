import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
	if len(W.shape) == 2:
		return cayley(W[None])[0]
	_, cout, cin = W.shape
	if cin > cout:
		return cayley(W.transpose(1, 2)).transpose(1, 2)
	U, V = W[:, :cin], W[:, cin:]
	I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
	A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
	iIpA = torch.inverse(I + A)
	return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)


def fft_shift_matrix(n, s):
	shift = torch.arange(0, n).repeat((n, 1))
	shift = shift + shift.T
	return torch.exp(1j * 2 * np.pi * s * shift / n)


class StridedConv(nn.Module):
	def __init__(self, *args, **kwargs):
		striding = False
		if 'stride' in kwargs and kwargs['stride'] == 2:
			kwargs['stride'] = 1
			striding = True
		super().__init__(*args, **kwargs)
		downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
		if striding:
			self.register_forward_pre_hook(lambda _, x: \
											   einops.rearrange(x[0], downsample, k1=2, k2=2))

class PaddingChannels(nn.Module):
	def __init__(self, ncin, ncout, scale=1.0):
		super().__init__()
		self.ncout = ncout
		self.ncin = ncin
		self.scale = scale

	def forward(self, x):
		bs, _, size1, size2 = x.shape
		out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
		out[:, :self.ncin] = self.scale * x
		return out

class PaddingFeatures(nn.Module):
	def __init__(self, fin, n_features, scale=1.0):
		super().__init__()
		self.n_features = n_features
		self.fin = fin
		self.scale = scale

	def forward(self, x):
		out = torch.zeros(x.shape[0], self.n_features, device=x.device)
		out[:, :self.fin] = self.scale * x
		return out

class PlainConv(nn.Conv2d):
	def forward(self, x):
		return super().forward(F.pad(x, (1, 1, 1, 1)))

class LinearNormalized(nn.Linear):

	def __init__(self, in_features, out_features, bias=True, scale=1.0):
		super(LinearNormalized, self).__init__(in_features, out_features, bias)
		self.scale = scale

	def forward(self, x):
		self.Q = F.normalize(self.weight, p=2, dim=1)
		return F.linear(self.scale * x, self.Q, self.bias)

class FirstChannel(nn.Module):
	def __init__(self, cout, scale=1.0):
		super().__init__()
		self.cout = cout
		self.scale = scale

	def forward(self, x):
		xdim = len(x.shape)
		if xdim == 4:
			return self.scale * x[:, :self.cout, :, :]
		elif xdim == 2:
			return self.scale * x[:, :self.cout]



######################### Sandwich layers [https://github.com/acfr/LBDN] #########################
class SandwichLin(nn.Linear):
	def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
		super().__init__(in_features+out_features, out_features, bias)
		self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
		self.alpha.data = self.weight.norm()
		self.scale = scale
		self.AB = AB
		self.Q = None

	def forward(self, x):
		fout, _ = self.weight.shape
		if self.training or self.Q is None:
			self.Q = cayley(self.alpha * self.weight / self.weight.norm())
		Q = self.Q if self.training else self.Q.detach()
		x = F.linear(self.scale * x, Q[:, fout:]) # B @ x
		if self.AB:
			x = 2 * F.linear(x, Q[:, :fout].T) # 2 A.T @ B @ x
		if self.bias is not None:
			x += self.bias
		return x

class SandwichFc(nn.Linear):
	def __init__(self, in_features, out_features, bias=True, scale=1.0):
		super().__init__(in_features+out_features, out_features, bias)
		self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
		self.alpha.data = self.weight.norm()
		self.scale = scale
		self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
		self.Q = None

	def forward(self, x):
		fout, _ = self.weight.shape
		if self.training or self.Q is None:
			self.Q = cayley(self.alpha * self.weight / self.weight.norm())
		Q = self.Q if self.training else self.Q.detach()
		x = F.linear(self.scale * x, Q[:, fout:]) # B*h
		if self.psi is not None:
			x = x * torch.exp(-self.psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
		if self.bias is not None:
			x += self.bias
		x = F.relu(x) * torch.exp(self.psi) # \Psi z
		x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
		return x

class SandwichConv1(nn.Module):
	def __init__(self,cin, cout, scale=1.0) -> None:
		super().__init__()
		self.scale = scale
		self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
		self.bias = nn.Parameter(torch.empty(cout))
		nn.init.xavier_normal_(self.kernel)
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
		bound = 1 / np.sqrt(fan_in)
		nn.init.uniform_(self.bias, -bound, bound)
		self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
		self.alpha.data = self.kernel.norm()
		self.Q = None

	def forward(self, x):
		cout = self.kernel.shape[0]
		if self.training or self.Q is None:
			P = cayley(self.alpha * self.kernel / self.kernel.norm())
			self.Q = 2 * P[:, :cout].T @ P[:, cout:]
		Q = self.Q if self.training else self.Q.detach()
		x = F.conv2d(self.scale * x, Q[:,:, None, None])
		x += self.bias[:, None, None]
		return F.relu(x)

class SandwichConv1Lin(nn.Module):
	def __init__(self, cin, cout, scale=1.0) -> None:
		super().__init__()
		self.scale = scale
		self.kernel = nn.Parameter(torch.empty(cout, cin + cout))
		self.bias = nn.Parameter(torch.empty(cout))
		nn.init.xavier_normal_(self.kernel)
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
		bound = 1 / np.sqrt(fan_in)
		nn.init.uniform_(self.bias, -bound, bound)
		self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
		self.alpha.data = self.kernel.norm()
		self.Q = None

	def forward(self, x):
		cout = self.kernel.shape[0]
		if self.training or self.Q is None:
			P = cayley(self.alpha * self.kernel / self.kernel.norm())
			self.Q = 2 * P[:, :cout].T @ P[:, cout:]
		Q = self.Q if self.training else self.Q.detach()
		x = F.conv2d(self.scale * x, Q[:, :, None, None])
		x += self.bias[:, None, None]
		return x

class SandwichConvLin(StridedConv, nn.Conv2d):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if 'stride' in kwargs and kwargs['stride'] == 2:
			args = list(args)
			args[0] = 4 * args[0]  # 4x in_channels
			if len(args) == 3:
				args[2] = max(1, args[2] // 2)  # //2 kernel_size; optional
				kwargs['padding'] = args[2] // 2  # TODO: added maxes recently
			elif 'kernel_size' in kwargs:
				kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
				kwargs['padding'] = kwargs['kernel_size'] // 2
		scale = 1.0
		if 'scale' in kwargs:
			scale = kwargs['scale']
			del kwargs['scale']
		args[0] += args[1]
		args = tuple(args)
		super().__init__(*args, **kwargs)
		self.scale = scale
		self.register_parameter('alpha', None)
		self.Qfft = None

	def forward(self, x):
		x = self.scale * x
		cout, chn, _, _ = self.weight.shape
		cin = chn - cout
		batches, _, n, _ = x.shape
		if not hasattr(self, 'shift_matrix'):
			s = (self.weight.shape[2] - 1) // 2
			self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n // 2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

		if self.training or self.Qfft is None or self.alpha is None:
			wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, chn,
																					n * (n // 2 + 1)).permute(2, 0,
																											  1).conj()
			if self.alpha is None:
				self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
			self.Qfft = cayley(self.alpha * wfft / wfft.norm())

		Qfft = self.Qfft if self.training else self.Qfft.detach()
		# Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
		xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
		xfft = 2 * Qfft[:, :, :cout].conj().transpose(1, 2) @ Qfft[:, :, cout:] @ xfft
		x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
		if self.bias is not None:
			x += self.bias[:, None, None]

		return x

class SandwichConv(StridedConv, nn.Conv2d):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if 'stride' in kwargs and kwargs['stride'] == 2:
			args = list(args)
			args[0] = 4 * args[0]  # 4x in_channels
			if len(args) == 3:
				args[2] = max(1, args[2] // 2)  # //2 kernel_size; optional
				kwargs['padding'] = args[2] // 2  # TODO: added maxes recently
			elif 'kernel_size' in kwargs:
				kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
				kwargs['padding'] = kwargs['kernel_size'] // 2
		scale = 1.0
		if 'scale' in kwargs:
			scale = kwargs['scale']
			del kwargs['scale']
		args[0] += args[1]
		args = tuple(args)
		super().__init__(*args, **kwargs)
		self.psi = nn.Parameter(torch.zeros(args[1]))
		self.scale = scale
		self.register_parameter('alpha', None)
		self.Qfft = None

	def forward(self, x):
		x = self.scale * x
		cout, chn, _, _ = self.weight.shape
		cin = chn - cout
		batches, _, n, _ = x.shape
		if not hasattr(self, 'shift_matrix'):
			s = (self.weight.shape[2] - 1) // 2
			self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n // 2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

		if self.training or self.Qfft is None or self.alpha is None:
			wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, chn,
																					n * (n // 2 + 1)).permute(2, 0,
																											  1).conj()
			if self.alpha is None:
				self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
			self.Qfft = cayley(self.alpha * wfft / wfft.norm())

		Qfft = self.Qfft if self.training else self.Qfft.detach()
		# Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
		xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
		xfft = 2 ** 0.5 * torch.exp(-self.psi).diag().type(xfft.dtype) @ Qfft[:, :, cout:] @ xfft
		x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
		if self.bias is not None:
			x += self.bias[:, None, None]
		xfft = torch.fft.rfft2(F.relu(x)).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cout, batches)
		xfft = 2 ** 0.5 * Qfft[:, :, :cout].conj().transpose(1, 2) @ torch.exp(self.psi).diag().type(xfft.dtype) @ xfft
		x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))

		return x



######################### Orthogonal layer [https://github.com/locuslab/orthogonal-convolutions] #########################
class OrthogonLin(nn.Linear):
	def __init__(self, in_features, out_features, bias=True, scale=1.0):
		super().__init__(in_features, out_features, bias)
		self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
		self.alpha.data = self.weight.norm()
		self.scale = scale
		self.Q = None

	def forward(self, x):
		if self.training or self.Q is None:
			self.Q = cayley(self.alpha * self.weight / self.weight.norm())
		Q = self.Q if self.training else self.Q.detach()
		y = F.linear(self.scale * x, Q, self.bias)
		return y

class OrthogonFc(nn.Linear):
	def __init__(self, in_features, out_features, bias=True, scale=1.0):
		super().__init__(in_features, out_features, bias)
		self.activation = nn.ReLU(inplace=False)
		self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
		self.alpha.data = self.weight.norm()
		self.scale = scale
		self.Q = None

	def forward(self, x):
		if self.training or self.Q is None:
			self.Q = cayley(self.alpha * self.weight / self.weight.norm())
		Q = self.Q if self.training else self.Q.detach()
		y = F.linear(self.scale * x, Q, self.bias)
		y = self.activation(y)
		return y

class OrthogonConvLin(StridedConv, nn.Conv2d):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if 'stride' in kwargs and kwargs['stride'] == 2:
			args = list(args)
			args[0] = 4 * args[0]  # 4x in_channels
			if len(args) == 3:
				args[2] = max(1, args[2] // 2)
				kwargs['padding'] = args[2] // 2
			elif 'kernel_size' in kwargs:
				kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
				kwargs['padding'] = kwargs['kernel_size'] // 2
		scale = 1.0
		if 'scale' in kwargs:
			scale = kwargs['scale']
			del kwargs['scale']
		args = tuple(args)
		super().__init__(*args, **kwargs)
		self.scale = scale
		self.register_parameter('alpha', None)
		self.Qfft = None

	def forward(self, x):
		x = self.scale * x
		cout, cin, _, _ = self.weight.shape
		batches, _, n, _ = x.shape
		if not hasattr(self, 'shift_matrix'):
			s = (self.weight.shape[2] - 1) // 2
			self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n // 2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
		xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
		if self.training or self.Qfft is None or self.alpha is None:
			wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin,
																					n * (n // 2 + 1)).permute(2, 0,
																											  1).conj()
			if self.alpha is None:
				self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
			self.Qfft = cayley(self.alpha * wfft / wfft.norm())
		Qfft = self.Qfft if self.training else self.Qfft.detach()
		yfft = (Qfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
		y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
		if self.bias is not None:
			y += self.bias[:, None, None]
		return y

class OrthogonConv(StridedConv, nn.Conv2d):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if 'stride' in kwargs and kwargs['stride'] == 2:
			args = list(args)
			args[0] = 4 * args[0]  # 4x in_channels
			if len(args) == 3:
				args[2] = max(1, args[2] // 2)
				kwargs['padding'] = args[2] // 2
			elif 'kernel_size' in kwargs:
				kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
				kwargs['padding'] = kwargs['kernel_size'] // 2
		scale = 1.0
		if 'scale' in kwargs:
			scale = kwargs['scale']
			del kwargs['scale']
		args = tuple(args)
		super().__init__(*args, **kwargs)
		self.scale = scale
		self.activation = nn.ReLU(inplace=False)
		self.register_parameter('alpha', None)
		self.Qfft = None


######################### LipRes layers #########################
class LipResFC(nn.Module):
	def __init__(self, nin, nout, L=1., activation=nn.ReLU):
		super().__init__()
		self.W = nn.Parameter(torch.randn(nin + nout, nout), requires_grad=True)
		torch.nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
		self.lamb = nn.Parameter(torch.randn(nout), requires_grad=True)
		self.b = nn.Parameter(torch.zeros(nout), requires_grad=True)
		self.sigma = activation()
		self.L = L

	def forward(self, x):
		x = x.T
		C = cayley(self.W)
		Lamb = 0.5 + torch.exp(self.lamb)

		n, nout = C.shape
		GT, HT = C[:nout, :], C[nout:, :]
		W = -1 * self.L / Lamb[:, None] * (GT @ HT.T)
		y = self.L * HT.T @ x + GT.T @ self.sigma(W @ x + self.b[:, None])

		return y.T

class LipResConv(nn.Module):
	def __init__(self, cin, cout, kernel_size, strided=True, L=1., activation=nn.ReLU):
		super().__init__()
		if strided:
			cin = 4 * cin
			kernel_size = max(1, kernel_size // 2)
			downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
			self.register_forward_pre_hook(lambda _, x: \
				                               einops.rearrange(x[0], downsample, k1=2, k2=2))

		self.cin = cin
		self.cout = cout
		self.k = kernel_size
		self.shift_amount = (self.k - 1) // 2
		self.strided = strided

		self.W = nn.Parameter(torch.randn(cin + cout, cout, self.k, self.k), requires_grad=True)
		torch.nn.init.xavier_uniform_(self.W)
		self.lamb = nn.Parameter(torch.randn(cout), requires_grad=True)
		self.b = nn.Parameter(torch.zeros(cout), requires_grad=True)
		self.sigma = activation()
		self.L = L
		self.shift_matrix = None

	def forward(self, x):
		batch, c, n, _ = x.shape
		nf = n // 2 + 1  # number of non-redundant fourier coefficients

		if self.shift_matrix is None:
			self.shift_matrix = fft_shift_matrix(n, -self.shift_amount).to(x.device)[:, :nf] \
				.reshape(n * nf, 1, 1)

		xfft = torch.fft.rfft2(x) \
			.permute(2, 3, 1, 0) \
			.reshape(n * nf, self.cin, batch)

		wfft = torch.fft.rfft2(self.W, (n, n)) \
			.reshape(self.cout + self.cin, self.cout, n * nf) \
			.permute(2, 0, 1).conj()  # [n * nf, cout + cin, cout]
		wfft = self.shift_matrix * wfft

		GHTs = cayley(wfft) # [n * nf, cin + cout, cout]
		GTs, HTs = GHTs[:, :self.cout, :], GHTs[:, self.cout:, :]  # [n * nf, cout, cout]
		GTHs = torch.bmm(GTs, HTs.transpose(1, 2).conj())  # [nfft, cout, cout]

		# compute the affine function in \sigma
		Lamb = 0.5 + torch.exp(self.lamb)
		GTHx_fft = torch.bmm(GTHs, xfft)  # [n * nf, cout, batch]
		GTHx = GTHx_fft.permute(2, 1, 0)  # [batch, cout, nfft]
		GTHx = GTHx.reshape(batch, self.cout, n, (n // 2 + 1))  # [batch, cout, n, n]
		GTHx = torch.fft.irfft2(GTHx)  # [batch, cout, n, n]
		affine = -1 * self.L / Lamb[None, :, None, None] * GTHx + self.b[None, :, None, None]  # [batch, cout, n, n]

		# compute G \sigma( ... )
		sig = self.sigma(affine)  # [batch, cout, n, n]
		sig_fft = torch.fft.rfft2(sig, (n, n))  # [batch, cout, n, nf]
		sig_fft = sig_fft.reshape(batch, self.cout, n * nf)  # [batch, cout, n  *nf]
		sig_fft = sig_fft.permute(2, 1, 0)  # [n * nf, cout, batch]
		Gsig_fft = torch.bmm(GTs.transpose(1, 2).conj(), sig_fft)  # [n * nf, cout, batch]
		Gsig = Gsig_fft.permute(2, 1, 0)  # [batch, cout, n * nf]
		Gsig = Gsig.reshape(batch, self.cout, n, nf)  # [batch, cout, n, n]
		Gsig = torch.fft.irfft2(Gsig)  # [batch, cout, n, n]

		# compute the residual connection Hx
		Hx_fft = torch.bmm(HTs.transpose(1, 2).conj(), xfft)  # [n * nf, cout, batch]
		Hx = Hx_fft.permute(2, 1, 0)  # [batch, cout, n * nf]
		Hx = Hx.reshape(batch, self.cout, n, nf)  # [batch, cout, n, nf]
		Hx = torch.fft.irfft2(Hx)  # [batch, cout, n, n]

		y = self.L * Hx + Gsig

		return y

class MultiMargin(nn.Module):

	def __init__(self, margin = 0.5):
		super().__init__()
		self.margin = margin

	def __call__(self, outputs, labels):
		return F.multi_margin_loss(outputs, labels, margin=self.margin)


if __name__ == '__main__':
	# liplinear = LipResFC(5, 10)
	# x = torch.randn(2, 5)
	# y = liplinear(x)
	# print(y.shape)
	# print(nn.Linear(3, 5).bias.shape)

	lipconv = LipResConv(3, 5, 3, strided=False)
	x = torch.randn(2, 3, 32, 32)
	y = lipconv(x)
	print(y.shape)
