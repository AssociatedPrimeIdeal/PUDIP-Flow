import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.PReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_z = x2.size()[-3] - x1.size()[-3]
        diff_y = x2.size()[-2] - x1.size()[-2]
        diff_x = x2.size()[-1] - x1.size()[-1]

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D_PUDIPFlow(nn.Module):
    def __init__(self, n_channels, n_classes, ft, depth=4):
        super().__init__()
        assert depth >= 1

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        factor = 2

        self.inc = DoubleConv(n_channels, ft)

        self.downs = nn.ModuleList()
        in_ch = ft
        for i in range(depth - 1):
            out_ch = ft * (2 ** (i + 1))
            self.downs.append(Down(in_ch, out_ch))
            in_ch = out_ch

        bottom_out_ch = ft * (2 ** depth) // factor
        self.bottom = Down(in_ch, bottom_out_ch)

        self.ups = nn.ModuleList()
        up_in_ch = ft * (2 ** depth)
        for i in reversed(range(depth)):
            out_ch = (ft * (2 ** i)) // (factor if i != 0 else 1)
            self.ups.append(Up(up_in_ch, out_ch))
            up_in_ch = ft * (2 ** i)

        self.outc = OutConv(ft, n_classes)

    def forward(self, x):
        x = self.inc(x)
        skips = [x]

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottom(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.outc(x)