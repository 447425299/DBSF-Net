import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt  # PyWavelets library for wavelet transforms

class WaveletDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, wavelet='haar', n_layers=3, norm_layer=nn.BatchNorm2d):
        super(WaveletDiscriminator, self).__init__()
        self.wavelet = wavelet
        self.ndf = ndf
        self.n_layers = n_layers

        # Define the wavelet transform layer
        self.wavelet_transform = WaveletTransform(wavelet)

        # Build the discriminator network
        self.build_discriminator(input_nc, ndf, n_layers, norm_layer)

    def build_discriminator(self, input_nc, ndf, n_layers, norm_layer):
        kw = 4
        padw = 1

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        # Intermediate convolutional layers
        layers = []
        for n in range(1, n_layers):
            layers += [
                nn.Conv2d(ndf * 2**(n-1), ndf * 2**n, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * 2**n),
                nn.LeakyReLU(0.2, True)
            ]

        # Final convolutional layer
        layers += [
            nn.Conv2d(ndf * 2**(n_layers-1), 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(
            self.conv1,
            self.leaky_relu,
            *layers
        )

    def forward(self, input):
        # Apply wavelet transform to the input image
        wavelet_coeffs = self.wavelet_transform(input)

        # Sum the coefficients across channels (assuming the input is a grayscale image)
        wavelet_sum = torch.sum(wavelet_coeffs, dim=1, keepdim=True)

        # Feed the summed wavelet coefficients through the discriminator network
        output = self.model(wavelet_sum)

        return output

    # output1 = discriminator1(input)
    # output2 = discriminator2(input)
    #
    # # 使用权重进行加权平均
    # weight1, weight2 = 0.7, 0.3  # 可以根据实际情况调整权重
    # combined_output = weight1 * output1 + weight2 * output2


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='haar'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, input):
        # Apply 2D wavelet transform
        coeffs = pywt.dwt2(input.cpu().numpy(), self.wavelet)
        LL, (LH, HL, HH) = coeffs

        # Stack the coefficients as channels
        wavelet_coeffs = torch.from_numpy(np.stack([LL, LH, HL, HH], axis=1)).to(input.device)

        return wavelet_coeffs