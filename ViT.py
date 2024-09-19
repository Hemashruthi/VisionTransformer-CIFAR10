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
    
# Utility function to calculate positional embeddings
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
        attn_output = attn_output.transpose(1, 2).view(B, N, E)

        print(f"Attention output shape: {attn_output.shape}")
        output = self.fc(attn_output)
        return output
    
if __name__ == "__main__":

  plt.imshow(get_positional_embeddings(100, 300), interpolation="nearest")
  plt.show()
  
  # Example input (Batch of images)
  img_size = 32 
  patch_size = 4
  in_channels = 3
  embed_dim = 128
  num_heads = 4
  
  x = torch.randn(16, in_channels, img_size, img_size)  # (batch_size, channels, height, width)

  # Patch Embedding
  patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
  patch_output = patch_embedding(x)

  # Positional Embeddings
  pos_embeddings = get_positional_embeddings((img_size // patch_size) ** 2, embed_dim)

  # Multi-head Attention
  attention = Attention(embed_dim, num_heads)
