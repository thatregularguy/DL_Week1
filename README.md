# DL_Week1
In DiH_ML.py I used the threshold calculated during the DiH group task. It uses 4 layer network.
Given Train and Test datas are concatenated and then spiltted after shuffling, just to decrease the risk of imbalance.
During few tries the network's prediction's MCC score was about 0.250 - 0.380.

In Network.py I created a DenseNetwork, which can do both regression and classification.
The only thing to be careful is the ReLU activation, which can cause calculation errors.
