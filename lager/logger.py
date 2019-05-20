from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass
class Config:
  buf_cap: int = 1000
  record_freq: int = 50
  plot_freq: int = 100


class Logger:
  def __init__(self, *labels, config=None):
    self.labels = labels
    if config is None:
      config = Config()
    self.config = config
    self.wins = defaultdict(lambda: None)
    self.clear()

  @property
  def num_labels(self):
    return len(self.labels)

  def _new_buf(self):
    return np.zeros((self.config.buf_cap, self.num_labels))

  def __getitem__(self, query):
    if isinstance(query, int):
      return self.entries[query]
    elif isinstance(query, str):
      return self.entries[:self._i, self.labels.index(query)]
    raise ValueError(f'invalid query type: {type(query)}')

  def record(self, *entry):
    if self._t % self.config.record_freq == 0:
      assert len(entry) == len(self.labels)
      if self._i == self.entries.shape[0]:
        self.entries = np.vstack((self.entries, self._new_buf()))
      self.entries[self._i] = np.array(entry)
      self._i += 1
    self._t += 1

  def plot(self, viz):
    if self._t % self.config.plot_freq == 0:
      for label in self.labels:
        x = np.arange(0, self._i, self.config.record_freq)
        y = self.entries[label]
        self.wins[label] = viz.line(X=x, Y=y, win=self.wins[label], opts=dict(
          title=label,
          xtick=True,
          ytick=True,
          xlabel='step',
          ylabel=label,
        ))

  def peek(self, top=10):
    print(', '.join(self.labels))
    for i in range(min(top, self._i)):
      print(', '.join([str(e) for e in self.entries[i]]))

  def clear(self):
    self.entries = self._new_buf()
    self._i = 0  # internal pointer
    self._t = 0  # internal counter

  def export(self, filename):
    with open(filename, 'w') as f:
      f.write(','.join(self.labels)+'\n')
      for entry in self.entries[:self._i]:
        f.write(','.join([str(n) for n in entry])+'\n')