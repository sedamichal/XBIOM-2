import torch
import torch.nn as nn

from TrainingFramewrok import Training


class TransformerModel(nn.Module):
    def __init__(
        self, input_size=1, model_dim=32, num_heads=4, num_layers=2, output_size=1
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq, model_dim)
        out = self.transformer(x)  # (batch, seq, model_dim)
        out = out[:, -1, :]  # poslední časový krok
        return self.fc(out)


class TransformerTraining(Training):
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
