import json
import os
import time
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.pipeline import BaseRolloutStore


class TestRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training SAC
    """

    def __init__(self):
        super().__init__()

        self.history = []

    def push(self, exps: int):
        self.history.append(exps)

    def clear_history(self):
        del self.history[:100]

    # def export_history(self, location: str):
    #     assert os.path.exists(location)

    #     fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

    #     def exp_to_dict(exp):
    #         {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

    #     data = [exp_to_dict(exp) for exp in self.history]
    #     with open(fpath, "w") as f:
    #         f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int):
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=shuffle)


store = TestRolloutStorage()
for i in range(200):
    store.push(i)
dataloader = store.create_loader(20, True)

for batch in dataloader:
    print(batch)
store.clear_history()
print("Cleared history")
for i in range(200, 300):
    store.push(i)
for batch in dataloader:
    print(batch)
