import torch
import torch.nn.functional as F


class AddandNorm(torch.nn.Module):
    """This torch module class performs the add and norm layer specified in the transformer encoder/decoder block"""
    def __init__(self):
        super().__init__()

    def operate(self, t):
        var, mean = torch.var_mean(t, dim = -1, keepdim = True)
        result = (t - mean)/torch.sqrt(var + 1e-5)
        return result

    def forward(self, x, z):
        return self.operate(x + z)


class FeedForwardNetwork(torch.nn.Module):
    """This torch module class performs the feedforward network block of the transformer encoder/decoder block with the
    specified hidden size."""
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.W1 = torch.nn.Linear(n_features, n_hidden)
        self.W2 = torch.nn.Linear(n_hidden, n_features)

    def forward(self, x):
        assert x.shape[-1] == self.W1.in_features, "The feature sizes do not match in the feed forward network"
        return self.W2(torch.nn.functional.relu(self.W1(x)))


class MaskedMultiheadAttention(torch.nn.Module):
    """This torch module is the attention block that is used in transformer blocks(both encoder and decoder) but
    only self attention which is not masked is happening in the Encoder and masked self attention and cross attention
    are present in the decoder. This class has the option of performing masked self attention as well."""
    def __init__(self, n_features, n_heads, masked = False):
        super().__init__()
        assert n_features % n_heads == 0, "The heads do not divide the features completely"
        self.Wq = torch.nn.Linear(n_features, n_features)
        self.Wk = torch.nn.Linear(n_features, n_features)
        self.Wv = torch.nn.Linear(n_features, n_features)
        self.n_heads = n_heads
        self.n_features = n_features
        self.masked = masked

    def splitheads(self, tensor):
        """This function splits the given tensor features (usually the last dimension) into separate heads of attention
        and transposes the context length and number of heads dimensions, for helping in matrix multiplication of
        queries, keys, and finally, values."""
        B, T, C = tensor.shape
        return tensor.contiguous().view(B, T, self.n_heads, -1).transpose(1,2)

    def mask(self, T):
        """This function returns a boolean mask of given dimension to help with the masking of self attention block
        before performing the softmax calculation"""
        return torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0).bool()

    def forward(self, X):
        B, T, C = X.shape
        Q = self.splitheads(self.Wq(X))  # batch size, heads, context length, features//heads
        K = self.splitheads(self.Wk(X))
        V = self.splitheads(self.Wv(X))
        scores = Q @ K.transpose(-1, -2)  # batch size, heads, context length, context length
        scores /= (self.n_features//self.n_heads) ** (0.5)
        if self.masked:
            scores = scores.masked_fill(~self.mask(T), float('-inf'))
        scores = F.softmax(scores, dim = -1)
        attn = scores @ V  # batch size, heads, context length, features//heads
        return attn.transpose(1,2).contiguous().view(B, T, -1)  # batch size, context length, features


class Embeddings(torch.nn.Module):
    """This torch module class adds the positional and feature embeddings of the characters before sending it into the
    transformer block."""
    def __init__(self, n_features, n_vocab, maxlen):
        super().__init__()
        self.E = torch.nn.Embedding(n_vocab, n_features)
        pe = torch.zeros((maxlen, n_features))
        num = torch.arange(0, maxlen).view(-1, 1)
        even_den = (10000 ** (torch.arange(0, n_features)[::2]/n_features))**(-1)  # same values as odd positions, size: (int(n_features/2))
        odd_den = (10000 ** (2*(torch.arange(0, n_features)[1::2]//2)/n_features)) ** (-1)  # same values as even positions, size: (n_features//2)
        pe[:, ::2] = torch.sin(num * even_den)  # size: maxlen, n_features/2
        pe[:, 1::2] = torch.cos(num * odd_den)  # size: maxlen, n_features//2
        pe = pe.unsqueeze(0)  # size: 1,maxlen,n_features
        self.register_buffer('pe', pe)  # pe tensor should not be learned as it is constant.

    def forward(self, x):
        assert x.shape[-1] == self.pe.shape[-2],"The context lengths do not match"  # for strictness, removing this can access context lengths less than maxlen
                                                                                    # and gives error for lengths greater than maxlen
        return self.E(x) + self.pe[:, :x.shape[1], :]  # pe will broadcast for batch size of E(x)


class EncoderBlock(torch.nn.Module):
    """This torch module builds the entire encoder block with one self attention block layer and dropout layers are added to
    avoid overfitting."""
    def __init__(self, n_features, n_heads, n_hidden, n_vocab):
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.multiheadattn = MaskedMultiheadAttention(n_features, n_heads)
        self.addnorm1 = AddandNorm()
        self.dropout1 = torch.nn.Dropout(0.1)
        self.feedforward = FeedForwardNetwork(n_features, n_hidden)
        self.addnorm2 = AddandNorm()
        self.dropout2 = torch.nn.Dropout(0.1)
        self.linearfinal = torch.nn.Linear(n_features, n_vocab)  # finally projecting onto the vocabulary space

    def forward(self, X):
        Z = self.multiheadattn(X)
        Z = self.addnorm1(X, Z)
        Z = self.dropout1(Z)
        A = self.feedforward(Z)
        return self.linearfinal(self.dropout2(self.addnorm2(Z, A)))


