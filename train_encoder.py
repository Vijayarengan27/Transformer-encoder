import encoderblock
import dataset
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# The training loop

g = torch.manual_seed(2147483647)
data = dataset.ReversedData(6)  # context length is 6
train_data, valid_data = random_split(data, [450, 50], generator=g)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator=g)  # training data in batches
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True, generator=g)  # validation data in batches

e = encoderblock.Embeddings(10, 27, 6)  # embeddings have 10 features with (26 alphabets + 1 padding) unique characters
eb = encoderblock.EncoderBlock(10, 2, 20, 27)  # transformer has 2 heads and hidden linear layer size is 20
parameters = list(e.parameters()) + list(eb.parameters())  # training the embeddings too (but not positional embeddings)
optimizer = torch.optim.Adam(parameters, lr=0.005)
print("The total number of learnable parameter values in this network is,", sum(p.numel() for p in parameters if p.requires_grad))

n_vocab = 27  # to flatten the logits at the end of encoder for loss calculation
train_losses, valid_losses = [], []
for epochs in range(100):
    train_loss, valid_loss = 0, 0
    eb.train()
    for batch_x, batch_y in train_loader:
        # forward pass
        optimizer.zero_grad()
        E = e(batch_x)
        logits = eb(E)
        logits_flat = logits.view(-1, n_vocab)
        y_flat = batch_y.view(-1)
        # loss calculation
        loss = torch.nn.functional.cross_entropy(logits_flat, y_flat)
        train_loss += loss.item()
        # backward pass
        loss.backward()
        # update
        optimizer.step()
    train_losses.append(train_loss/len(train_loader))

    with torch.no_grad():
        eb.eval()  # to remove dropout layer while validating
        for x, y in valid_loader:
            E = e(x)
            logits = eb(E)
            logits_flat = logits.view(-1, n_vocab)
            y_flat = y.view(-1)
            # loss calculation
            loss = torch.nn.functional.cross_entropy(logits_flat, y_flat)
            valid_loss += loss.item()
        valid_losses.append(valid_loss/len(valid_loader))


# plot the losses

plt.plot(train_losses, label="Training loss")
plt.plot(valid_losses, label="Validation loss")
plt.legend()
plt.ylabel("Loss per epoch")
plt.xlabel("Epochs")
plt.title("Loss curve")
plt.savefig("images/loss_curve.png")  # to save it in the images folder

# Final: Inference, here add whatever word (six or fewer letters) and the model will reverse it

with torch.no_grad():
    eb.eval()  # this is done to deactivate the dropout layer
    reverse = torch.argmax(eb(e(data.encode('sushis').unsqueeze(0))), dim=-1)  # getting the required reversed characters from the highest values' index
    print(''.join(data.idx_to_chr[t.item()] for t in reverse.squeeze() if t.item() != 0))

