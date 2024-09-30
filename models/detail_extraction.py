import torch
import torch.nn as nn

class INNBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(INNBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # device = x.device
        # x = x.detach().cpu()
        conv1 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, padding=self.kernel_size // 2).to(x.device)
        conv2 = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, padding=self.kernel_size // 2).to(x.device)
        norm = nn.InstanceNorm2d(self.input_channels).to(x.device)
        act = nn.ReLU()

        y = conv1(x)
        y = act(y)
        y = conv2(y)
        y = norm(y)
        y = x + y
        return y

class DetailFeatureExtractor(nn.Module):
    def __init__(self, num_blocks, hidden_channels, kernel_size):
        super(DetailFeatureExtractor, self).__init__()
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # 获取输入特征的通道数
        input_channels = x.size(1)

        # 根据输入特征的通道数创建INNBlock
        blocks = nn.ModuleList([
            INNBlock(input_channels, self.hidden_channels, self.kernel_size)
            for _ in range(self.num_blocks)
        ])

        # 处理输入特征
        for block in blocks:
            x = block(x)

        return x

# class DetailFeatureExtractor(nn.Module):
#     def __init__(self, input_channels_list, hidden_channels, kernel_size):
#         super(DetailFeatureExtractor, self).__init__()
# 
#         self.inn_blocks = nn.ModuleList([INNBlock(channels, hidden_channels, kernel_size) for channels in input_channels_list])
# 
#     def forward(self, x_list):
#         out_list = [inn_block(x) for inn_block, x in zip(self.inn_blocks, x_list)]
#         return out_list