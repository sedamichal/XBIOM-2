import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score
import torch.nn.functional as F
import datetime
import numpy as np
from DataSet import MultiOutputTimeSeriesDataset
import os
import json


class TrainingLog:
    def __init__(self, log_path):
        self._log_path = log_path
        self._log = dict()
        self._current_epoch = 0
        self.current_epoch_start = None

    def log_start(self):
        self._log["start"] = datetime.datetime.now().isoformat()
        self._log["epochs"] = dict()
        print(f"Training started at {self._log['start']}")
        # print("\tEpoch\tTraining:\tloss\tMSE\tMAE\tR2\tValidation:\tloss\tMSE\tMAE\tR2")

    def log_end(self):
        self._log["end"] = datetime.datetime.now().isoformat()
        print(f"Training finished at {self._log['end']}")

        with open(self._log_path, "w") as fp:
            json.dump(self._log, fp)

    def log_file(self, file):
        self._log["file"] = file
        print(f"Model saved in {self._log['file']}")

    def log_epoch_start(self, epoch):
        self._current_epoch = epoch
        self._log["epochs"][epoch] = dict()
        self._log["epochs"][epoch]["start"] = datetime.datetime.now().isoformat()
        self._current_epoch_start = datetime.datetime.now()
        # print(f"Epoch {epoch} started at {self._log['epochs'][epoch]['start']}")

    def log_epoch_end(
        self,
        epoch,
        train_loss,
        train_mse,
        train_mae,
        train_r2,
        val_loss,
        val_mse,
        val_mae,
        val_r2,
    ):
        self._log["epochs"][epoch]["train"] = dict()
        self._log["epochs"][epoch]["train"]["loss"] = train_loss
        self._log["epochs"][epoch]["train"]["mse"] = train_mse
        self._log["epochs"][epoch]["train"]["mae"] = train_mae
        self._log["epochs"][epoch]["train"]["r2"] = train_r2
        self._log["epochs"][epoch]["val"] = dict()
        self._log["epochs"][epoch]["val"]["loss"] = val_loss
        self._log["epochs"][epoch]["val"]["mse"] = val_mse
        self._log["epochs"][epoch]["val"]["mae"] = val_mae
        self._log["epochs"][epoch]["val"]["r2"] = val_r2
        self._log["epochs"][epoch]["end"] = datetime.datetime.now().isoformat()
        self._current_epoch_start = None
        self._current_epoch = None
        print(
            "\tEpoch {0:3d}\tTraining: loss: {1:4.4f}\tMSE: {2:4.4f}\tMAE: {3:2.4f}\tR2: {4:2.4f}\tValidation: loss: {5:4.4f}\tMSE: {6:4.4f}\tMAE: {7:2.4f}\tR2: {8:2.4f}".format(
                epoch,
                train_loss,
                train_mse,
                train_mae,
                train_r2,
                val_loss,
                val_mse,
                val_mae,
                val_r2,
            )
        )


class Training:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        training_dataset_list,
        validation_dataset_list,
        log_path,
        model_path,
        target_cols,
        batch_size=16,
        seq_len=30,
    ):
        self._model = model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._optimizer = optimizer
        self._criterion = criterion
        self._training_dataset_list = training_dataset_list
        self._validation_dataset_list = validation_dataset_list
        self._log = None
        self._log_folder = log_path
        self._model_path = model_path
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._target_cols = target_cols
        self._predictions = []
        self._targets = []

    def run(self, epochs):
        start = datetime.datetime.now()
        self._log = TrainingLog(self._log_path(self._log_folder, start))
        self._log.log_start()
        best_val_loss = 99999.99

        for epoch in range(epochs):
            train_metrics = np.zeros(4)
            val_metrics = np.zeros(4)
            total_train_samples = 0
            total_val_samples = 0

            self._log.log_epoch_start(epoch)

            self._model.train()
            for seq in self._training_dataset_list:
                if len(seq) < self._seq_len:
                    continue
                ds = MultiOutputTimeSeriesDataset(
                    seq, seq_len=self._seq_len, target_cols=self._target_cols
                )
                loader = DataLoader(ds, batch_size=self._batch_size, shuffle=True)
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self._device)
                    batch_y = batch_y.to(self._device)
                    loss, mse, mae, r2 = self._run_train_batch(batch_x, batch_y)
                    total_train_samples += batch_y.size(0)  # počet vzorků v batch
                    train_metrics += np.array(
                        [loss.item(), mse, mae, r2]
                    ) * batch_y.size(0)
            train_metrics /= total_train_samples
            train_metrics[3] = self._get_r2()

            self._model.eval()
            with torch.no_grad():
                for seq in self._validation_dataset_list:
                    if len(seq) < self._seq_len:
                        continue
                    ds = MultiOutputTimeSeriesDataset(
                        seq, seq_len=self._seq_len, target_cols=self._target_cols
                    )
                    loader = DataLoader(ds, batch_size=self._batch_size, shuffle=False)
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(self._device)
                        batch_y = batch_y.to(self._device)
                        loss, mse, mae, r2 = self._run_val_batch(batch_x, batch_y)
                        total_val_samples += batch_y.size(0)  # počet vzorků v batch
                        val_metrics += np.array(
                            [loss.item(), mse, mae, r2]
                        ) * batch_y.size(0)
                val_metrics /= total_val_samples

                if best_val_loss > val_metrics[0]:
                    best_val_loss = val_metrics[0]
                    self._save_model(start)

                val_metrics[3] = self._get_r2()

            self._log.log_epoch_end(
                epoch,
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                train_metrics[3],
                val_metrics[0],
                val_metrics[1],
                val_metrics[2],
                val_metrics[3],
            )

        self._log.log_end()

    def _get_r2(self):
        buff_preds = torch.cat(self._predictions, 0).cpu()
        buff_targets = torch.cat(self._targets, 0).cpu()
        self._predictions = []
        self._targets = []
        return r2_score(buff_preds, buff_targets).item()

    def _run_val_batch(self, batch_x, batch_y):
        out = self._model(batch_x)

        self._predictions.append(out.detach())
        self._targets.append(batch_y.detach())

        loss = self._criterion(out, batch_y)
        mse = self._mse(out, batch_y)
        mae = self._mae(out, batch_y)
        r2 = 0  # r2_score(out, batch_y).item()
        return loss, mse, mae, r2

    def _run_train_batch(self, batch_x, batch_y):
        self._optimizer.zero_grad()
        out = self._model(batch_x)

        self._predictions.append(out.detach())
        self._targets.append(batch_y.detach())

        loss = self._criterion(out, batch_y)
        loss.backward()
        self._optimizer.step()

        mse = self._mse(out, batch_y)
        mae = self._mae(out, batch_y)
        r2 = 0
        return loss, mse, mae, r2

    def _save_model(self, timestamp):
        path = self._state_path(timestamp)
        torch.save(self._model.state_dict(), path)

    def _state_path(self, timestamp):
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

        return os.path.join(
            self._model_path, f"{timestamp.strftime('%Y-%m-%d-%H-%M-%SZ')}.pth"
        )

    def _log_path(self, log_path, timestamp):
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        return os.path.join(
            log_path, f"{timestamp.strftime('%Y-%m-%d-%H-%M-%SZ')}.json"
        )

    def _mse(self, pred, batch_y):
        return F.mse_loss(pred, batch_y, reduction="mean").item()

    def _mae(self, pred, batch_y):
        return F.l1_loss(pred, batch_y, reduction="mean").item()
