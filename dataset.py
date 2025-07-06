import torch
import string


class ReversedData(torch.utils.data.Dataset):
    """This is a torch dataset class that gives random words of length maxlen -3 to maxlen - 1 for given data size.
     Then this also reverses the words which are its targets. This data is then used as a toy dataset for the transformer
     block to train on."""
    def __init__(self, maxlen, size=500):
        self.size = size
        self.maxlen = maxlen
        self.data, self.target, self.chr_to_idx, self.idx_to_chr = self._create_data()

    def _create_data(self):
        """This function creates the data upon initialization of an instance of this class"""
        a = [w for w in string.ascii_lowercase]
        # make character index dictionary for lookup
        chr_to_idx = {ch:i+1 for i,ch in enumerate(string.ascii_lowercase)}
        idx_to_chr = {i+1:ch for i,ch in enumerate(string.ascii_lowercase)}
        idx_to_chr[0] = '< >'  # padding for encoding upto maxlen characters
        chr_to_idx['< >'] = 0
        g = torch.manual_seed(2147483647)  # for reproducibility
        sizes = torch.randint(self.maxlen-3, self.maxlen, (self.size,),generator=g)
        data, target = [], []
        for j in range(sizes.shape[0]):
            word = ''
            for i in range(sizes[j].item()):
                x = torch.randint(1,27,(1,), generator=g).item()
                word += idx_to_chr[x]
            data.append(word)
        for w in data:
            target.append(w[::-1])  # simple reversal
        return data, target, chr_to_idx, idx_to_chr

    def encode(self, word, reverse=False):
        """Here we are encoding the word to its index with padding upto given maxlen."""
        encoded_word = [self.chr_to_idx[w] for w in word[:self.maxlen]]
        padding = [0] * (self.maxlen - len(encoded_word))
        if reverse:
            return torch.tensor(padding + encoded_word)
        return torch.tensor(encoded_word + padding)

    def decode(self, tensor):
        """Decode the letters alone without the padding"""
        return ''.join([self.idx_to_chr[tensor[t].item()] for t in range(tensor.shape[0]) if tensor[t].item() != 0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.encode(self.data[idx])
        y = self.encode(self.target[idx], reverse=True)
        return x, y