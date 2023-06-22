class cells_net(nn.Module):
  def __init__(self):
    super(cells_net, self).__init__()
    num_input_channels = 1
    self.net_arch = nn.Sequential(
        nn.Conv2d(num_input_channels, 8, kernel_size=3), # 1x256x256 => 8x254x254
        nn.ReLU(),
        nn.MaxPool2d(2), # 8x254x254 => 8x127x127

        nn.Conv2d(8, 16, kernel_size=3), # 8x127x127 => 16x125x125
        nn.ReLU(),
        nn.MaxPool2d(2), # 16x125x125 => 16x62x62

        nn.Conv2d(16, 16, kernel_size=3), # 16x62x62 => 16x60x60
        nn.ReLU(),
        nn.MaxPool2d(2), # 16x60x60 => 16x30x30

        nn.Conv2d(16, 32, kernel_size=3), # 16x30x30 => 32x28x28
        nn.ReLU(),
        nn.MaxPool2d(2), # 32x28x28 => 32x14x14

        nn.Conv2d(32, 64, kernel_size=3), # 32x14x14 => 64x12x12
        nn.ReLU(),
        nn.MaxPool2d(2), # 64x12x12 => 64x6x6

        nn.Flatten(), # 64x6x6 => 2304
        nn.Linear(2304,256),
        nn.ReLU(),
        nn.Linear(256,2),
        nn.Sigmoid()
    )

  def forward(self, x):
      out = self.net_arch(x)
      return out
