import torch

class Dataloader:

    def __init__(self, data, batch_size, window_size):
        super().__init__()
        self.B = batch_size
        self.T = window_size
        self.data = torch.tensor(data, dtype=torch.long)
        self.start = 0

    def reset(self):
        self.start = 0

    def next_batch(self):
        length = self.B * self.T
        batch_data = self.data[self.start: self.start + length + 1]
        x = batch_data[:-1].view(self.B, self.T)
        labels = batch_data[1:].view(self.B, self.T)

        self.start += length
        if self.start + length >= len(self.data):
            self.reset()

        return x, labels