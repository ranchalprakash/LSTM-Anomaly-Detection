# LSTM-Anomaly-Detection
LSTM Network for anomaly detection of a sinusoidal wave

The dataset is a timeseries dataset that comprises of a combination of sinusoisal signals, to make it easier, the signals will have three frequencies:

1. 10Hz(y1)
2. 30Hz(y2)
3. 50Hz(y3)

The dataset looks something like:

{x: <milli_second_value>; y: <value of the summation of the three signals = value of y1 + value of y2 + value of y3>}

Model generates these signals and the dataset and detect anomalies that arecintroduced in the signal. 


