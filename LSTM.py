import torch
import torch.nn as nn

from TrainingFramewrok import Training


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMTraining(Training):
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
