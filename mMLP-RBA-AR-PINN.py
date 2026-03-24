import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
from tqdm import trange

torch.cuda.empty_cache()  # 释放内存
import numpy as np
import scipy.io

torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

''' 定义各类激活函数 '''

class NewAdaptiveSinActivation(nn.Module):
    def __init__(self, init_amplitude=1.0, init_phase=0.0):
        super(NewAdaptiveSinActivation, self).__init__()
        self.amplitude = nn.Parameter(torch.tensor(init_amplitude, dtype=torch.float32))
        self.phase = nn.Parameter(torch.tensor(init_phase, dtype=torch.float32))

    def get_parameters(self):
        """返回当前的参数值"""
        return {'amplitude': self.amplitude.item(), 'phase': self.phase.item()}

    def forward(self, input):
        # a_o * sqrt(2) * sin(x + p_s + π/4)
        return self.amplitude * np.sqrt(2) * torch.sin(input + self.phase + np.pi/4)

class SinActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class CosActivation(nn.Module):
    def forward(self, input):
        return torch.cos(input)

class AtanActivation(nn.Module):
    def forward(self, input):
        return torch.atan(input)

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.full((1,), 0.5), requires_grad=True)
        self.w2 = nn.Parameter(torch.full((1,), 0.5), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

# 在已有的激活函数类之后添加
class AdaptiveSinActivation(nn.Module):
    def __init__(self, init_amplitude=1.0, init_phase=0.0):
        super(AdaptiveSinActivation, self).__init__()
        # 定义可学习参数
        self.amplitude = nn.Parameter(torch.tensor(init_amplitude))
        self.phase = nn.Parameter(torch.tensor(init_phase))

    def get_parameters(self):
        """返回当前的参数值"""
        return {
            'amplitude': self.amplitude.item(),
            'phase': self.phase.item(),
        }

    def forward(self, input):
        return self.amplitude * torch.sin( input + self.phase)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[20, 20, 5], activation='tanh'):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Softplus())

        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Softplus())

        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

class MscaleDNN_2(nn.Module):
    def __init__(self, input_size: int, hidden_size: list, output_size: int, scale_factors: list,
                 activation: str = 'sin', dropout_rate=0.0):
        super(MscaleDNN_2, self).__init__()
        self.scale_factors = scale_factors
        self.subnetworks = nn.ModuleList()

        self.activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'atan': AtanActivation(),
            'sin': SinActivation(),
            'wave':WaveAct(),
        }

        for scale in self.scale_factors:
            layers = OrderedDict()
            layers['input_layer'] = nn.Linear(input_size, hidden_size[0])
            layers.update(self._build_hidden_layers(hidden_size, activation))
            layers['output_layer'] = nn.Linear(hidden_size[-1], output_size)
            self.subnetworks.append(nn.Sequential(layers))

    def _build_hidden_layers(self, hidden_size: list, activation: str, dropout_rate=0.0) -> OrderedDict:
        layers = OrderedDict()
        for i in range(1, len(hidden_size)):
            layers[f'hidden_layer_{i}'] = nn.Linear(hidden_size[i - 1], hidden_size[i])
            layers[f'activation_{i}'] = self.activations.get(activation, SinActivation())
            if dropout_rate > 0:
                layers[f'dropout_{i}'] = nn.Dropout(p=dropout_rate)
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = sum(subnetwork(x * scale) for subnetwork, scale in zip(self.subnetworks, self.scale_factors))
        return outputs

class DenseResNet(nn.Module):
    def __init__(self, dim_in=3, dim_out=2, num_resnet_blocks=3,
                 num_layers_per_block=2, num_neurons=40, activation='sin',
                 fourier_features=True, m_freqs=256, sigma=4.6, tune_beta=False):
        super(DenseResNet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = self._get_activation(activation)
        self.tune_beta = tune_beta

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))
        else:
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.encoder_U = nn.Sequential(
            nn.Linear(dim_in, num_neurons),
            self.activation
        )

        self.encoder_V = nn.Sequential(
            nn.Linear(dim_in, num_neurons),
            self.activation
        )

        self.first = nn.Linear(dim_in, num_neurons)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons)
                           for _ in range(num_layers_per_block)])
            for _ in range(num_resnet_blocks)])

        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_resnet_blocks)])

        self.last = nn.Linear(num_neurons, dim_out)

        if fourier_features:
            self.first = nn.Linear(2 * m_freqs, num_neurons)
            # self.B = nn.Parameter(sigma * torch.randn(dim_in, m_freqs))  # to converts inputs to m_freqs
            self.B = sigma * torch.randn(dim_in, m_freqs)
            # 生成从 -8 到 8 之间的 128 个等间隔的数
            # row = torch.linspace(-3, 3, steps=128)
            # # 将其扩展为 2 行 128 列的张量
            # self.B = row.unsqueeze(0).repeat(2, 1)

    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'atan':
            return AtanActivation()
        elif name == 'sin':
            return SinActivation()
        elif name == 'cos':
            return CosActivation()
        elif name == 'wave':
            return WaveAct()
        elif name == 'adaptive_sin':
            return AdaptiveSinActivation()
        elif name == 'new_adaptive_sin':
            return NewAdaptiveSinActivation()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, x):
        x = x.to(device)
        self.beta0 = self.beta0.to(device)
        self.beta = self.beta.to(device)

        U = self.encoder_U(x)
        V = self.encoder_V(x)

        if self.fourier_features:
            self.B = self.B.to(device)
            cosx = torch.cos(torch.matmul(x, self.B))
            sinx = torch.sin(torch.matmul(x, self.B))
            x = torch.cat((cosx, sinx), dim=1)
            x = self.activation(self.beta0 * self.first(x))
        else:
            x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))

            z = (1 - z) * U + z * V
            x = self.alphas[i] * z + (1 - self.alphas[i]) * x

        out = self.last(x)
        return out

    def model_capacity(self):
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print("\n\nThe number of layers in the model: %d" % num_layers)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    def get_alphas(self):
        return [alpha.item() for alpha in self.alphas]

# 均方误差 MSE
mse_loss = torch.nn.MSELoss()

def auto_grad(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
    else:
        return auto_grad(auto_grad(u, x), x, order=order - 1)

def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

# 随机初始化（均匀分布）的函数
def random_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, a=-0.8, b=0.8)  # 均匀分布初始化，范围为[-1, 1]
        nn.init.zeros_(layer.bias)  # 初始化偏置为零

def lecun_init(layer):
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.size(1)  # 获取输入特征数量
        std = 1.0 / np.sqrt(fan_in)    # Lecun初始化的标准差
        nn.init.normal_(layer.weight, mean=0.0, std=std)  # 使用正态分布初始化权重
        nn.init.zeros_(layer.bias)      # 将偏置初始化为零


from torch.optim import lr_scheduler


class PINNs():
    def __init__(self, layers, activation, device=device, initial_lr=0.001, sadap=False, is_MscaleDNN=False,
                 use_resnet=False, resnet_params=None, rba=False):
        self.device = device
        self.train_loss = []
        self.is_MscaleDNN = is_MscaleDNN
        self.use_resnet = use_resnet
        self.alphas_history = [[] for _ in range(resnet_params['num_resnet_blocks'])] if use_resnet else []
        self.rba = rba
        self.eta = 0.01
        self.gamma = 0.999

        # 记录权重
        self.weight_max_history = []
        self.weight_min_history = []
        self.weight_mean_history = []

        # 添加快照相关的属性
        self.snapshot_epochs = [100, 500, 2000, 4000, 8000, 10000, 15000, 20000, 30000]
        self.weights_snapshots = {}
        self.residuals_snapshots = {}

        if is_MscaleDNN:
            self.dnn = MscaleDNN_2(
                input_size=layers[0], output_size=layers[-1], hidden_size=layers[1:-1],
                scale_factors=[1, 2, 4, 8, 16, 32],
                activation=activation,
            ).to(device)
        elif use_resnet:
            assert resnet_params is not None, "ResNet parameters must be provided when use_resnet=True"
            self.dnn = DenseResNet(
                dim_in=layers[0], dim_out=layers[-1], num_resnet_blocks=resnet_params['num_resnet_blocks'],
                num_layers_per_block=resnet_params['num_layers_per_block'], num_neurons=resnet_params['num_neurons'],
                activation=activation
            ).to(device)
        else:
            self.dnn = NeuralNetwork(
                input_size=layers[0], output_size=layers[-1], hidden_size=layers[1:-1],
                activation=activation,
            ).to(device)

        self.dnn.apply(xavier_init)

        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.sadap = sadap
        self.optimizer_Adam = torch.optim.Adam(params=self.dnn.parameters(), lr=initial_lr, weight_decay=0.000)
        if self.sadap:
            self.scheduler = lr_scheduler.StepLR(self.optimizer_Adam, step_size=1000, gamma=0.9)

        self.iter = 0
        self.residual_weights_A = torch.ones(x_train.shape[0],1, device=self.device)
        self.residual_weights_B = torch.ones(x_train.shape[0],1, device=self.device)


    def rba_update(self, residual, residual_weights):
        norm_residual = torch.abs(residual) / torch.max(torch.abs(residual))
        residual_weights = (self.gamma * residual_weights + self.eta * norm_residual).detach()
        return residual_weights

    def closure(self):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()

        loss = self.lossf(
            model=self.dnn,
            td=None,
            print_str="",
        )
        self.iter += 1
        if loss.requires_grad:
            loss.backward()

        self.train_loss.append(loss.item())

        s = f'迭代次数: {self.iter} Loss: {loss.item():.6f}'
        if self.iter % 10 == 0:
            print(s)

        return loss

    def train(self, epochs, lossf):
        self.dnn.train()
        self.lossf = lossf

        with trange(epochs, dynamic_ncols=True, ncols=1) as td:
            for epoch in td:
                self.dnn.train()
                loss = self.lossf(model=self.dnn, td=td, print_str="")
                self.train_loss.append(loss.item())
                self.optimizer_Adam.zero_grad()
                loss.backward()
                self.optimizer_Adam.step()

                if self.sadap:
                    self.scheduler.step()

                if self.use_resnet:
                    alphas = self.dnn.get_alphas()
                    for i, alpha in enumerate(alphas):
                        self.alphas_history[i].append(alpha)

                if self.rba:
                    # 计算残差
                    residual_A = self.compute_residual_A(self.dnn)
                    residual_B = self.compute_residual_B(self.dnn)

                    # 更新权重
                    self.residual_weights_A = self.rba_update(residual_A, self.residual_weights_A)
                    self.residual_weights_B = self.rba_update(residual_B, self.residual_weights_B)

                    # 每1000轮打印状态
                    if epoch % 1000 == 0:
                        print(f"\nEpoch {epoch} Status:")
                        print(f"Residual A - min: {residual_A.min():.4f}, max: {residual_A.max():.4f}")
                        print(f"Current weights A - min: {self.residual_weights_A.min():.4f}, "
                              f"max: {self.residual_weights_A.max():.4f}, "
                              f"mean: {self.residual_weights_A.mean():.4f}")

                    # 记录权重历史
                    self.weight_max_history.append(torch.max(self.residual_weights_A).item())
                    self.weight_min_history.append(torch.min(self.residual_weights_A).item())
                    self.weight_mean_history.append(torch.mean(self.residual_weights_A).item())

                # 在指定epoch保存快照
                if (epoch + 1) in self.snapshot_epochs:
                    print(f"\nSaving snapshot at epoch {epoch + 1}")
                    self.save_snapshot(epoch + 1)

            # 保存权重历史
            scipy.io.savemat('weight_history.mat', {
                'max_weights': self.weight_max_history,
                'min_weights': self.weight_min_history,
                'mean_weights': self.weight_mean_history
            })

            # 保存快照数据
            scipy.io.savemat('all_snapshots.mat', {
                'weights_snapshots': self.weights_snapshots,
                'residuals_snapshots': self.residuals_snapshots,
                'snapshot_epochs': self.snapshot_epochs
            })

            self.optimizer.step(closure=self.closure)

            # 保存训练损失
            network_type = "MscaleDNN" if self.is_MscaleDNN else "ResNetPINN" if self.use_resnet else "PINN"
            scipy.io.savemat(f"{network_type}_final_train_loss_new_adaptive_sin.mat", {'train_loss': self.train_loss})

            if self.use_resnet:
                for i, alphas in enumerate(self.alphas_history):
                    scipy.io.savemat(f"{network_type}_alpha_{i + 1}_history.mat", {f'alpha_{i + 1}_history': alphas})

    def save_snapshot(self, epoch):
        """在特定epoch保存权重和残差的快照"""
        if epoch in self.snapshot_epochs:
            # 计算残差
            residual_A = self.compute_residual_A(self.dnn)  # 实部残差
            residual_B = self.compute_residual_B(self.dnn)  # 虚部残差

            # 保存当前状态
            weights_A_np = self.residual_weights_A.detach().cpu().numpy()  # 实部权重
            weights_B_np = self.residual_weights_B.detach().cpu().numpy()  # 虚部权重
            residual_A_np = residual_A.detach().cpu().numpy()
            residual_B_np = residual_B.detach().cpu().numpy()

            # 打印形状检查
            print(f"\nSnapshot at epoch {epoch}:")
            print(f"Weights A shape: {weights_A_np.shape}")  # 应该是(15000,)
            print(f"Weights B shape: {weights_B_np.shape}")  # 应该是(15000,)
            print(f"Residuals A shape: {residual_A_np.shape}")  # 应该是(15000,)
            print(f"Residuals B shape: {residual_B_np.shape}")  # 应该是(15000,)

            # 分别保存实部和虚部的统计信息
            print("\nReal part statistics:")
            print(
                f"Weights A - min: {np.min(weights_A_np):.4f}, max: {np.max(weights_A_np):.4f}, mean: {np.mean(weights_A_np):.4f}")
            print(
                f"Residuals A - min: {np.min(residual_A_np):.4f}, max: {np.max(residual_A_np):.4f}, mean: {np.mean(residual_A_np):.4f}")

            print("\nImaginary part statistics:")
            print(
                f"Weights B - min: {np.min(weights_B_np):.4f}, max: {np.max(weights_B_np):.4f}, mean: {np.mean(weights_B_np):.4f}")
            print(
                f"Residuals B - min: {np.min(residual_B_np):.4f}, max: {np.max(residual_B_np):.4f}, mean: {np.mean(residual_B_np):.4f}")

            # 保存到mat文件，清晰区分实部和虚部的数据
            scipy.io.savemat(f'snapshot_{epoch}.mat', {
                'weights_real': weights_A_np,  # 实部权重
                'weights_imag': weights_B_np,  # 虚部权重
                'residuals_real': residual_A_np,  # 实部残差
                'residuals_imag': residual_B_np,  # 虚部残差
                'epoch': epoch,
                'x_train': x_train,
                'z_train': z_train
            })

            print(f"Snapshot at epoch {epoch} saved successfully")

    def predict(self, X, batch_size=1000):
        """
        分批预测以减少内存使用
        """
        self.dnn.eval()
        self.dnn.to(device)

        # 准备存储所有预测结果的数组
        n_samples = X.shape[0]
        dUr_pred_list = []
        dUi_pred_list = []
        dUr_dxx_list = []
        dUr_dzz_list = []
        dUi_dxx_list = []
        dUi_dzz_list = []

        # 分批处理
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i + batch_size]

            x_tensor = torch.from_numpy(batch_X[:, 0:1]).float().to(device).requires_grad_(True)
            z_tensor = torch.from_numpy(batch_X[:, 1:2]).float().to(device).requires_grad_(True)

            inputs = torch.cat([x_tensor, z_tensor], dim=1)
            outputs = self.dnn(inputs)

            dUr_pred = outputs[:, :1]
            dUi_pred = outputs[:, 1:2]

            # 计算导数
            dUr_dx = auto_grad(dUr_pred, x_tensor, 1)
            dUr_dz = auto_grad(dUr_pred, z_tensor, 1)
            dUi_dx = auto_grad(dUi_pred, x_tensor, 1)
            dUi_dz = auto_grad(dUi_pred, z_tensor, 1)

            dUr_dxx = auto_grad(dUr_dx, x_tensor, 1)
            dUr_dzz = auto_grad(dUr_dz, z_tensor, 1)
            dUi_dxx = auto_grad(dUi_dx, x_tensor, 1)
            dUi_dzz = auto_grad(dUi_dz, z_tensor, 1)

            # 转换为numpy并存储
            dUr_pred_list.append(dUr_pred.cpu().detach().numpy())
            dUi_pred_list.append(dUi_pred.cpu().detach().numpy())
            dUr_dxx_list.append(dUr_dxx.cpu().detach().numpy())
            dUr_dzz_list.append(dUr_dzz.cpu().detach().numpy())
            dUi_dxx_list.append(dUi_dxx.cpu().detach().numpy())
            dUi_dzz_list.append(dUi_dzz.cpu().detach().numpy())

            # 清理不需要的张量
            torch.cuda.empty_cache()

        # 合并所有批次的结果
        dUr_pred = np.vstack(dUr_pred_list)
        dUi_pred = np.vstack(dUi_pred_list)
        dUr_dxx = np.vstack(dUr_dxx_list)
        dUr_dzz = np.vstack(dUr_dzz_list)
        dUi_dxx = np.vstack(dUi_dxx_list)
        dUi_dzz = np.vstack(dUi_dzz_list)

        return (dUr_pred, dUi_pred, dUr_dxx, dUr_dzz, dUi_dxx, dUi_dzz)

    def compute_residual_A(self, model):
        x_tensor = torch.from_numpy(x_train).float().to(device).requires_grad_(True)
        z_tensor = torch.from_numpy(z_train).float().to(device).requires_grad_(True)
        m_tensor = torch.from_numpy(m_train).float().to(device)
        m0_tensor = torch.from_numpy(m0_train).float().to(device)
        U0_real_tensor = torch.from_numpy(U0_real_train).float().to(device)

        inputs = torch.cat([x_tensor, z_tensor], dim=1)
        predictions = model(inputs)
        dUr = predictions[:, :1]

        dUr_x = auto_grad(dUr, x_tensor, 1)
        dUr_xx = auto_grad(dUr_x, x_tensor, 1)
        dUr_z = auto_grad(dUr, z_tensor, 1)
        dUr_zz = auto_grad(dUr_z, z_tensor, 1)

        residual = (w ** 2) * m_tensor * dUr + dUr_xx + dUr_zz + (w ** 2) * (m_tensor - m0_tensor) * U0_real_tensor
        return residual

    def compute_residual_B(self, model):
        x_tensor = torch.from_numpy(x_train).float().to(device).requires_grad_(True)
        z_tensor = torch.from_numpy(z_train).float().to(device).requires_grad_(True)
        m_tensor = torch.from_numpy(m_train).float().to(device)
        m0_tensor = torch.from_numpy(m0_train).float().to(device)
        U0_imag_tensor = torch.from_numpy(U0_imag_train).float().to(device)

        inputs = torch.cat([x_tensor, z_tensor], dim=1)
        predictions = model(inputs)
        dUi = predictions[:, 1:2]

        dUi_x = auto_grad(dUi, x_tensor, 1)
        dUi_xx = auto_grad(dUi_x, x_tensor, 1)
        dUi_z = auto_grad(dUi, z_tensor, 1)
        dUi_zz = auto_grad(dUi_z, z_tensor, 1)

        residual = (w ** 2) * m_tensor * dUi + dUi_xx + dUi_zz + (w ** 2) * (m_tensor - m0_tensor) * U0_imag_tensor
        return residual

def pinn_pde_lossA(model, loss=F.mse_loss):
    x_tensor = torch.from_numpy(x_train).float().to(device).requires_grad_(True)
    z_tensor = torch.from_numpy(z_train).float().to(device).requires_grad_(True)
    m_tensor = torch.from_numpy(m_train).float().to(device)
    m0_tensor = torch.from_numpy(m0_train).float().to(device)
    U0_real_tensor = torch.from_numpy(U0_real_train).float().to(device)

    inputs = torch.cat([x_tensor, z_tensor], dim=1)
    predictions = model(inputs)
    dUr = predictions[:, :1]

    dUr_x = auto_grad(dUr, x_tensor, 1)
    dUr_xx = auto_grad(dUr_x, x_tensor, 1)
    dUr_z = auto_grad(dUr, z_tensor, 1)
    dUr_zz = auto_grad(dUr_z, z_tensor, 1)

    zero_tensor = torch.zeros_like(dUr)

    loss_A = loss((w ** 2) * m_tensor * dUr + dUr_xx + dUr_zz + (w ** 2) * (m_tensor - m0_tensor) * U0_real_tensor,
         zero_tensor, reduction='none')

    return loss_A

def pinn_pde_lossB(model, loss=F.mse_loss):
    x_tensor = torch.from_numpy(x_train).float().to(device).requires_grad_(True)
    z_tensor = torch.from_numpy(z_train).float().to(device).requires_grad_(True)
    m_tensor = torch.from_numpy(m_train).float().to(device)
    m0_tensor = torch.from_numpy(m0_train).float().to(device)
    U0_imag_tensor = torch.from_numpy(U0_imag_train).float().to(device)

    inputs = torch.cat([x_tensor, z_tensor], dim=1)
    predictions = model(inputs)
    dUi = predictions[:, 1:2]

    dUi_x = auto_grad(dUi, x_tensor, 1)
    dUi_xx = auto_grad(dUi_x, x_tensor, 1)
    dUi_z = auto_grad(dUi, z_tensor, 1)
    dUi_zz = auto_grad(dUi_z, z_tensor, 1)

    zero_tensor = torch.zeros_like(dUi)

    loss_B = loss((w ** 2) * m_tensor * dUi + dUi_xx + dUi_zz + (w ** 2) * (m_tensor - m0_tensor) * U0_imag_tensor,
         zero_tensor, reduction='none')

    return loss_B

# def get_pinn_loss(model, td=None, print_str=""):
#     loss_A = pinn_pde_lossA(model=model)
#     loss_B = pinn_pde_lossB(model=model)
#
#     if pinn_resnet.rba:
#         loss_all = torch.mean(pinn_resnet.residual_weights_A * loss_A + pinn_resnet.residual_weights_B * loss_B)
#     else:
#         loss_all = loss_A + loss_B
#
#     if td is not None:
#         td.set_description(
#             f"实部: {loss_A.item():.5f}, 虚部: {loss_B.item():.5f}, 总的: {loss_all.item():.6f}" + print_str)
#
#     return loss_all

def get_pinn_loss(model, td=None, print_str=""):
    loss_A = pinn_pde_lossA(model=model)
    loss_B = pinn_pde_lossB(model=model)

    if pinn_resnet.rba:
        weighted_loss_A = pinn_resnet.residual_weights_A * loss_A
        weighted_loss_B = pinn_resnet.residual_weights_B * loss_B
        loss_all = torch.mean(weighted_loss_A + weighted_loss_B)
    else:
        weighted_loss_A = loss_A
        weighted_loss_B = loss_B
        loss_all = torch.mean(loss_A + loss_B)

    if td is not None:
        td.set_description(
            f"实部: {loss_A.mean().item():.5f}, 虚部: {loss_B.mean().item():.5f}, 加权后的实部: {weighted_loss_A.mean().item():.5f}, 加权后的虚部: {weighted_loss_B.mean().item():.5f}" + print_str)

    return loss_all


f = 26
w = 2 * np.pi * f

data_test_5Hz = scipy.io.loadmat('layer_26Hz_test_data.mat')
x_test = data_test_5Hz['x_star']
z_test = data_test_5Hz['z_star']
du_real_true = data_test_5Hz['dU_real_star']
du_imag_true = data_test_5Hz['dU_imag_star']

data_5Hz = scipy.io.loadmat('layer_26Hz_train_data.mat')
U0_real_train = data_5Hz['U0_real_train']
U0_imag_train = data_5Hz['U0_imag_train']
x_train = data_5Hz['x_train']
z_train = data_5Hz['z_train']
m_train = data_5Hz['m_train']
m0_train = data_5Hz['m0_train']

layers = [2, 2]
resnet_params = {
    'num_resnet_blocks': 3,
    'num_layers_per_block': 2,
    'num_neurons': 100,
}

pinn_resnet = PINNs(
    layers=layers,
    activation='sin',
    device=device,
    sadap=True, initial_lr=0.001,
    use_resnet=True,
    resnet_params=resnet_params,
    rba=False  # 启用RBA机制
)

pinn_resnet.train(epochs=20000, lossf=get_pinn_loss)

X_test = np.hstack((x_test, z_test))

predictions_resnet = pinn_resnet.predict(X_test, batch_size=1000)

dUr_pred_resnet, dUi_pred_resnet, dUr_pred_resnet_dxx, dUr_pred_resnet_dzz, dUi_pred_resnet_dxx, dUi_pred_resnet_dzz = predictions_resnet

scipy.io.savemat('du_real_pred_layer_resnet_26hz.mat', {'du_real_pred': dUr_pred_resnet})
scipy.io.savemat('du_imag_pred_layer_resnet_26hz.mat', {'du_imag_pred': dUi_pred_resnet})
scipy.io.savemat('auto_26hz.mat', {'dUr_dxx': dUr_pred_resnet_dxx,'dUr_dzz': dUr_pred_resnet_dzz,'dUi_dxx': dUi_pred_resnet_dxx,'dUi_dzz': dUi_pred_resnet_dzz})

print("训练结束，预测结果已保存")

# 计算 L2 误差
def compute_relative_l2_error(pred, true):
    # 计算误差
    error = pred - true
    # 计算 L2 绝对误差
    l2_error = np.sqrt(np.sum(error**2))
    # 计算真实值的 L2 范数
    l2_true = np.sqrt(np.sum(true**2))
    # 计算相对 L2 误差
    relative_l2_error = l2_error / l2_true
    return relative_l2_error

# 计算 dUr 和 dUi 的相对 L2 误差
relative_l2_error_dUr = compute_relative_l2_error(dUr_pred_resnet, du_real_true)
relative_l2_error_dUi = compute_relative_l2_error(dUi_pred_resnet, du_imag_true)

# 打印结果
print(f"相对 L2 误差 (dUr): {relative_l2_error_dUr:.6f}")
print(f"相对 L2 误差 (dUi): {relative_l2_error_dUi:.6f}")

print("训练结束，结果已保存")
print("sigma=4.6 26hz")


