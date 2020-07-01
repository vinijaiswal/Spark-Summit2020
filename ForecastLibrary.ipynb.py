# Databricks notebook source
# Python
import pandas as pd
import numpy as np
from fbprophet import Prophet
import seaborn


# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

# Python
df = pd.read_csv('https://github.com/stirlingw/fb-prophet-example/raw/master/data/example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
df.head()

# COMMAND ----------

# Python
m = Prophet()
m.fit(df);

# COMMAND ----------

# Python
future = m.make_future_dataframe(periods=365)
future.tail()

# COMMAND ----------

# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

# Python
m.plot(forecast);

# COMMAND ----------

# Python
m.plot_components(forecast);