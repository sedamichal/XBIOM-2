import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


class Result:
    def __init__(self, base_dir=r".\results", log_path="LOG"):
        self._root = base_dir
        self._log_path = log_path
        self._data = self._join_results()

    def _get_model_data(self, model_name, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            start = datetime.datetime.fromisoformat(data.get("start"))
            end = datetime.datetime.fromisoformat(data.get("end"))
            duration = end - start
            new_log = {
                "model": model_name,
                "start": start,
                "epochs": {},
                "end": end,
                "duration": duration.seconds / 60,  # divmod(duration.seconds, 60),
            }

            for epoch, epoch_data in data["epochs"].items():
                int_epoch = int(epoch)
                new_log["epochs"][int_epoch] = epoch_data["val"]

        return new_log

    def _join_results(self):
        all_data = []

        if os.path.isdir(self._root):
            for model_name in os.listdir(self._root):
                model_path = os.path.join(self._root, model_name)
                log_path = os.path.join(model_path, self._log_path)
                if os.path.isdir(log_path):
                    # najít všechny JSON soubory v LOG
                    json_files = sorted(glob.glob(os.path.join(log_path, "*.json")))
                    if not json_files:
                        continue

                    last_json = json_files[-1]

                    all_data.append(self._get_model_data(model_name, last_json))
        return all_data

    def show_metrics(self):
        self._show_loss()
        self._show_r2()
        # self._show_mse()
        # self._show_mae()
        self.show_time()

    def _get_metric_data(self, metric):
        x = []
        Y = dict()
        for model_data in self._data:
            if len(x) == 0:
                x = model_data["epochs"].keys()

            y = [e[metric] for e in model_data["epochs"].values()]
            Y[model_data["model"]] = y
        return x, Y

    def _show_metric(self, metric):
        x, Y = self._get_metric_data(metric)

        plt.figure(figsize=(8, 5))

        for model, vals in Y.items():
            plt.plot(x, vals, label=model)

        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} over epochs")
        plt.grid(True)
        plt.legend()
        plt.show()

    def _show_r2(self):
        self._show_metric("r2")

    def _show_mse(self):
        self._show_metric("mse")

    def _show_mae(self):
        self._show_metric("mae")

    def _show_loss(self):
        self._show_metric("loss")

    def show_time(self):
        x = [e["model"] for e in self._data]
        y = [e["duration"] for e in self._data]

        plt.figure(figsize=(8, 5))
        plt.bar(x, y)
        plt.ylabel("[Min]")
        plt.title("Duration")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    res = Result(base_dir=r".\RESULTS")
    res.show_metrics()
    res.show_time()
