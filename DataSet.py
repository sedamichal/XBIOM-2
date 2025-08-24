import torch
from torch.utils.data import Dataset


class MultiOutputTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=30, target_cols=["power12", "power3", "power4"]):
        self.seq_len = seq_len
        self.target_cols = target_cols

        # odstraneni casoveho pole ts
        self.features = df.drop(columns=["ts"]).values.astype("float32")
        # zjistíme indexy cílových sloupců v poli features
        self.target_idx = [
            df.columns.get_loc(c) - 1 for c in target_cols
        ]  # -1 protože ts je dropnuto

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]  # (seq_len, input_size)
        y = self.features[idx + self.seq_len, self.target_idx]  # (num_targets,)
        return torch.tensor(X), torch.tensor(y)
