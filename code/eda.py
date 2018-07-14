import pandas as pd
import numpy as np
import random
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from collections import Counter

df = pd.read_csv("../input/mitbih_test.csv", header=None)
print(df.shape)
print(Counter(df[187].values))

Y = np.array(df[187].values).astype(np.int8)
X = np.array(df[list(range(187))].values)

indexes = random.sample(list(range(df.shape[0])), 10)

for i in indexes:

    data = [go.Scatter(
              x=list(range(187)),
              y=X[i, :])]

    plot({"data": data,
          "layout": {"title": "Heartbeat Class : %s "%Y[i]}}, filename='%s.html'%i)
