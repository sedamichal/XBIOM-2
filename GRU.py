import torch
import torch.nn as nn

from TrainingFramewrok import Training


class MyGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Váhy pro jednu vrstvu
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev=None):
        """
        x: vstup sekvence tvaru (seq_len, batch, input_size)
        h_prev: počáteční hidden state (batch, hidden_size)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # zmena poradi na (batch, seq_len, input_size)

        batch_size, seq_len, _ = x.size()

        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )

        # jen jedna vrstva (num_layers = 1)
        h_t = h_prev[0]

        out = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_t))
            r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h_t))
            h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h_t))
            h_t = (1 - z_t) * h_t + z_t * h_tilde

            out.append(h_t)

        # spojeni do (batch, seq_len, hidden_size)
        out = torch.stack(out, dim=1)

        h_n = h_t.unsqueeze(0)

        if not self.batch_first:
            out = out.transpose(0, 1)  # zpatky na (seq_len, batch, hidden_size)

        return out, h_n


class MyGRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.gru = MyGRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # poslední časový krok
        return self.fc(out)


class MyGRUTraining(Training):
    def __init__(
        self,
        model,
        training_dataset_list,
        validation_dataset_list,
        log_path,
        model_path,
        target_cols,
        batch_size=16,
    ):
        super().__init__(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
            criterion=nn.MSELoss(),
            training_dataset_list=training_dataset_list,
            validation_dataset_list=validation_dataset_list,
            log_path=log_path,
            model_path=model_path,
            batch_size=batch_size,
            target_cols=target_cols,
        )


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # poslední časový krok
        return self.fc(out)


class GRUTraining(Training):
    def __init__(
        self,
        model,
        training_dataset_list,
        validation_dataset_list,
        log_path,
        model_path,
        target_cols,
        batch_size=16,
    ):
        super().__init__(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
            criterion=nn.MSELoss(),
            training_dataset_list=training_dataset_list,
            validation_dataset_list=validation_dataset_list,
            log_path=log_path,
            model_path=model_path,
            batch_size=batch_size,
            target_cols=target_cols,
        )
