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
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

# 2. Utility function to calculate positional embeddings
def get_positional_embeddings(num_patches, embed_dim):
    result = torch.ones(num_patches, embed_dim)
    for i in range(num_patches):
        for j in range(embed_dim):
            result[i][j] = np.sin(i / (10000 ** (j / embed_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / embed_dim)))
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
        Q = self.q_proj(x)  # (B, N, E)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split Q, K, V for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale and compute attention weights
        Q = Q * self.scale
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Attention scores
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention weights to the values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, E)
        output = self.fc(attn_output)
        return output

# Feed Forward Network
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

# Transformer Encoder Layer   
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
        attn_output = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_output)
        hidden_output = self.hidden(self.norm2(x))
        x = x + self.dropout2(hidden_output)
        return x

# Transformer Encoder
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

# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # CLS token
        self.pos_embed = nn.Parameter(get_positional_embeddings((img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0] #num of images in a batch
        x = self.patch_embed(x)  # (B, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)  # output x = (B , num_patches + 1, new_embed_dim)
        cls_output = x[:, 0]  # class token from the encoded x
        output = self.mlp_head(cls_output)
        return output

# Function to display images with labels in a grid
def imshow_grid(images, labels, predicted_labels=None):
    fig, axs = plt.subplots(8, 8, figsize=(12, 12))
    axs = axs.flatten()

    for i in range(64):
        img = images[i] / 2 + 0.5
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        if predicted_labels is None:
            axs[i].set_title(f'Label: {test_dataset.classes[labels[i]]}')
        else:
            axs[i].set_title(f'Label: {test_dataset.classes[labels[i]]}\nPrediction: {test_dataset.classes[predicted_labels[i]]}')
        axs[i].imshow(img)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Vision Transformer model
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        hidden_dim=32,
        num_layers=4,
        dropout=0.1
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy on the test set: {(100 * correct / total):.2f}%')

    # Display images and predictions
    images, labels = next(iter(test_loader))

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    imshow_grid(images, labels, predicted)
