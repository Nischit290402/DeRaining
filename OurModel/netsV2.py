import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


from modules import DilatedResidualBlock, NLB, DGNL, DepthWiseDilatedResidualBlock


# Helpers
def pair(t):
    return t if isinstance(t, tuple) else (t,t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# Main Classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head), 
                FeedForward(dim, mlp_dim)
            ]))
    
    def forward(self, x):
        # print("Transformer class' x: ",x.shape)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim)
        )

    def forward(self, img):
        # print("Even before",img.shape)
        
        *_, h, w, dtype = *img.shape, img.dtype
        
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        # print("Before",x.shape)
        x = rearrange(x, 'b h w c -> b (h w) c') + pe

        x = self.transformer(x)
        # print(x.shape)
        
        x = self.linear_head(x)
        x = rearrange(x, 'b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)', h1 = h//self.patch_height, w1 = w//self.patch_width, p1 = self.patch_height, p2 = self.patch_width)
        
        # x = x.mean(dim = 1)
        # print("After",x.shape)

        x = self.to_latent(x)
        return x



class DGNLNet_fast(nn.Module):
    def __init__(self, num_features=64):
        super(DGNLNet_fast, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 8, stride=4, padding=2),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        # self.conv5 = nn.Sequential(
		# 	nn.Conv2d(128, 128, 3, padding=2, dilation=2),
		# 	nn.GroupNorm(num_groups=32, num_channels=128),
		# 	nn.SELU(inplace=True)
		# )

        self.vit = SimpleViT(
            image_size=64,
            patch_size = 16,
            num_classes = 0,
            dim = 128,
            depth = 8,
            heads = 6,
            mlp_dim = 512,
            channels = 128
        )

        self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)


        self.depth_pred = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	############################################ Rain removal network


        self.head = nn.Sequential(
			# pw
			nn.Conv2d(3, 32, 1, 1, 0, 1, bias=False),
			nn.ReLU6(inplace=True),
			# dw
			nn.Conv2d(32, 32, kernel_size=8, stride=4, padding=2, groups=32, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(32, num_features, 1, 1, 0, 1, 1, bias=False),
		)

        self.body = nn.Sequential(
			DepthWiseDilatedResidualBlock(num_features, num_features, 1),
			# DepthWiseDilatedResidualBlock(num_features, num_features, 1),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			DepthWiseDilatedResidualBlock(num_features, num_features, 4),
			DepthWiseDilatedResidualBlock(num_features, num_features, 8),
			DepthWiseDilatedResidualBlock(num_features, num_features, 4),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			# DepthWiseDilatedResidualBlock(num_features, num_features, 1),
			DepthWiseDilatedResidualBlock(num_features, num_features, 1)
		)


        self.dgnlb = DGNL(num_features)


        self.tail = nn.Sequential(
			# dw
			nn.ConvTranspose2d(num_features, 32, kernel_size=8, stride=4, padding=2, groups=32, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(32, 3, 1, 1, 0, 1, bias=False),
		)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f5 = self.vit(d_f3)
        d_f8 = self.conv8(d_f5)
        d_f9 = self.conv9(d_f8 + d_f2)
        depth_pred = self.depth_pred(d_f9 + d_f1)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        if self.training:
            return x, F.interpolate(depth_pred, size=x.size()[2:], mode='bilinear', align_corners=True)

        return x



class DGNLNet(nn.Module):
    def __init__(self, num_features=64):
        super(DGNLNet, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        #---------------------------------------------- Depth prediction network --------------------------------------
        self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv4 = nn.Sequential(
			nn.Conv2d(128, 256, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv5 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=4, dilation=4),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)
		
        self.vit = SimpleViT(
            image_size=32,
            patch_size = 8,
            num_classes = 0,
            dim = 128,
            depth = 8,
            heads = 6,
            mlp_dim = 512,
            channels = 256
        )

        self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv10 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.depth_pred = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	#--------------------------------------------------- Rain removal network ------------------------------------------

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.dgnlb = DGNL(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        #------------------------------------------------- Depth prediction network ------------------------------------------
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5_7 = self.vit(d_f4)
        d_f8 = self.conv8(d_f5_7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)

        depth_pred = self.depth_pred(d_f10 + d_f1)
        #--------------------------------------------------- Rain removal network --------------------------------------------

        f = self.head(x)
        f = self.body(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        if self.training:
            return x, depth_pred

        return x


# device = torch.device("cuda:0")
#
# net = GDNLNet().to(device)
# summary(net, (3, 512, 1024))
