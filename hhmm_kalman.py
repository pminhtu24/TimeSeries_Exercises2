import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error

np.random.seed(242)
n_samples = 980
data = np.cumsum(np.random.randn(n_samples))

df = pd.DataFrame(data, columns=["Value"])

data_cleaned = (df - df.mean()) / df.std()

observations = data_cleaned.values

high_res_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
high_res_model.fit(data_cleaned)

low_res_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
low_res_model.fit(data_cleaned)

kf = KalmanFilter(dim_x=2, dim_z=1)

kf.F = np.array([[1, 1], [0, 1]])
kf.H = np.array([[1, 0]])
kf.P *= 1000
kf.R = 5
kf.Q = np.array([[0.001, 0], [0, 0.001]])

predictions = []
for z in observations:
    high_res_state = high_res_model.predict([z])[0]

    kf.predict()
    kf.update(z)

    predictions.append(kf.x[0])

plt.figure(figsize=(10, 6))
plt.plot(observations, label="Giá trị thực tế", color='blue')
plt.plot(predictions, label="Giá trị dự đoán (Kalman)", linestyle="--", color='red')
plt.legend()
plt.title('So sánh giá trị thực tế và giá trị dự đoán với Kalman Filter và HHMM')
plt.show()

mse = mean_squared_error(observations, predictions)
print(f"Mean Squared Error (MSE): {mse}")
