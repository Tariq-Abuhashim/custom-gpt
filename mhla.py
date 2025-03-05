import torch
import torch.nn as nn
import torch.nn.functional as F

# Step 1: Tokenization (using a simple word-level tokenizer)
def tokenize(text):
    return text.lower().split()

# Step 2: Vocabulary and Embedding
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)

# Step 3: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.encoding = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_seq_len, embed_dim)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

# Step 4: Multi-Head Attention (MHA)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformations for Q, K, V
        Q = self.query(query)  # (batch_size, seq_len, embed_dim)
        K = self.key(key)      # (batch_size, seq_len, embed_dim)
        V = self.value(value)  # (batch_size, seq_len, embed_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.fc_out(output)

        return output

# Step 5: Multi-Head Latent Attention (MHLA)
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads

        assert self.head_dim * num_heads == latent_dim, "Latent dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, latent_dim)
        self.key = nn.Linear(embed_dim, latent_dim)
        self.value = nn.Linear(embed_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformations for Q, K, V in latent space
        Q = self.query(query)  # (batch_size, seq_len, latent_dim)
        K = self.key(key)      # (batch_size, seq_len, latent_dim)
        V = self.value(value)  # (batch_size, seq_len, latent_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention in latent space
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        output = self.fc_out(output)  # Project back to original embedding dimension

        return output

# Step 6: Putting It All Together
class TextProcessor(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, latent_dim, max_seq_len):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.mhla = MultiHeadLatentAttention(embed_dim, num_heads, latent_dim)

    def forward(self, token_ids):
        # Step 1: Embed tokens
        x = self.embedding(token_ids)  # (batch_size, seq_len, embed_dim)
        
        # Step 2: Add positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_len, embed_dim)
        
        # Step 3: Apply MHA (self-attention)
        mha_output = self.mha(x, x, x)  # (batch_size, seq_len, embed_dim)
        
        # Step 4: Apply MHLA (self-attention in latent space)
        mhla_output = self.mhla(x, x, x)  # (batch_size, seq_len, embed_dim)
        
        return mha_output, mhla_output

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000  # Size of the vocabulary
    embed_dim = 64      # Embedding dimension
    num_heads = 8       # Number of attention heads
    latent_dim = 128    # Latent dimension for MHLA
    max_seq_len = 20    # Maximum sequence length

    # Example input text
    text = "Hello, world! This is a test."

    # Step 1: Tokenize
    tokens = tokenize(text)
    print("Tokens:", tokens)

    # Step 2: Convert tokens to token IDs (using a dummy vocabulary)
    vocab = {word: idx for idx, word in enumerate(set(tokens))}
    token_ids = torch.tensor([[vocab[token] for token in tokens]])  # (batch_size=1, seq_len)

    # Step 3: Process text through the model
    model = TextProcessor(vocab_size, embed_dim, num_heads, latent_dim, max_seq_len)
    mha_output, mhla_output = model(token_ids)

    print("MHA Output Shape:", mha_output.shape)  # (1, seq_len, embed_dim)
    print("MHLA Output Shape:", mhla_output.shape)  # (1, seq_len, embed_dim)
