import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 1. Patch Embedding 
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.proj(x)
        print(f"After projection: {x.shape}") # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)
        print(f"After flattening: {x.shape}") # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        print(f"After transposing: {x.shape}")
        return x
    
# 2. Utility function to calculate positional embeddings
def get_positional_embeddings(num_patches, embed_dim):
    print(f"Generating positional embeddings for {num_patches} patches and {embed_dim} dimensions.")
    result = torch.ones(num_patches, embed_dim)
    for i in range(num_patches):
        for j in range(embed_dim):
            result[i][j] = np.sin(i / (10000 ** (j / embed_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / embed_dim)))
    print(f"Positional embeddings shape: {result.shape}")
    return result

  
# Multi-Head Attention Class
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

         # Linear layers for projecting the input to queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, E = x.shape
        print(f"Input to attention: {x.shape}")
        Q = self.q_proj(x)  # (B, N, E)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split Q, K, V for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        # Scale and compute attention weights
        Q = Q * self.scale
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Attention scores
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        print(f"Attention scores shape: {attn_scores.shape}")  # Print attention scores shape
        print(f"Attention weights shape: {attn_weights.shape}")  # Print attention weights shape

        # Apply attention weights to the values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, E)

        print(f"Attention output shape: {attn_output.shape}")
        output = self.fc(attn_output)
        return output
    
# 4. Multi - layer Perception 
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# 5. Transformer Encoder Layers   
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.hidden = FeedForward(embed_dim, hidden_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention & skip connection
        attn_output = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_output)

        # MLP with skip connection
        hidden_output = self.hidden(self.norm2(x))
        x = x + self.dropout2(hidden_output)

        return x
    
# 6. Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
encoder = TransformerEncoder(embed_dim = 128, num_heads = 8, hidden_dim = 32, num_layers = 4, dropout = 0.1)
x = torch.randn(32, 16, 128)
output = encoder(x)
print(output.shape)

# 7. Vision Transforms
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #CLS token
        self.pos_embed = nn.Parameter(get_positional_embeddings((img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers, dropout )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0] #num of images in a batch
        x = self.patch_embed(x) # (B, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x) # output x = (B , num_patches + 1, new_embed_dim)
        cls_output = x[:, 0] # class token from the encoded x
        output = self.mlp_head(cls_output)
        return output
    
img_size = 32
patch_size = 4
in_channels = 3
num_classes = 10
embed_dim = 128
num_heads = 4
hidden_dim = 32
num_layers = 4
dropout = 0.1

vit = VisionTransformer(img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, hidden_dim, num_layers, dropout)
x = torch.randn(32, 3, 32, 32)  # (batch_size, channels, height, width)
output = vit(x)
print(output.shape)  # (Batch_size , num_classes)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
# if __name__ == "__main__":

#   plt.imshow(get_positional_embeddings(100, 300), interpolation="nearest")
#   plt.show()
  
#   # Example input (Batch of images)
#   img_size = 32 
#   patch_size = 4
#   in_channels = 3
#   embed_dim = 128
#   num_heads = 4
  
#   x = torch.randn(16, in_channels, img_size, img_size)  # (batch_size, channels, height, width)

#   # Patch Embedding
#   patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#   patch_output = patch_embedding(x)

#   # Positional Embeddings
#   pos_embeddings = get_positional_embeddings((img_size // patch_size) ** 2, embed_dim)

#   # Multi-head Attention
#   attention = Attention(embed_dim, num_heads)
