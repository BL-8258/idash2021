# FL-project

## Dependencies:
Tensorflow 2.3.0 <br>
Tensorflow-Privacy 0.5.2 <br>
Tensorflow-Federated 0.17.0  <br>

## Introduction
The model is based on tensorflow_federated and tensorflow_privacy frameworks. There are two parties and a central server in this program. There are two parts of codes: train.py and predict.py for training and predicting process respectively. There are two dataset files (party_1.csv, party_2.csv) in the “Dataset” directory that contain the training data for two parties. The structure of the network is shown

<!--- ![LSTM2](https://user-images.githubusercontent.com/111822855/188915049-a88f1249-fdc5-41e9-a7f0-31d0ce32ec81.png) -->

<div align="center">
<img src="https://user-images.githubusercontent.com/111822855/188915049-a88f1249-fdc5-41e9-a7f0-31d0ce32ec81.png" width="500" height="400" alt="Model"/><br/>
</div>


## Tutorials
Please make sure the checkpoint_manager.py is on the same directory with train.py and predict.py.
For training process, you can use Command-Line Interface like following command:

```
python train.py --rounds=30 --epochs=20 --batch_size=30 --noise_multiplier=1.0 --model_dir='M_NAME'
```
Once the training process is finished, the trained model would be saved on the path of M_NAME.

The parameters you can set are:


<div align="center">

<table id="tfhover" class="tftable" border="1">
<tr><th>Parameters</th><th>Default</th><th>Comment</th></tr>
<tr><td>noise_multiplier</td><td>1.2</td><td>Ratio of the standard deviation to the clipping norm</td></tr>
<tr><td>batch_size</td><td>30</td><td>Batch size</td></tr>
<tr><td>epochs</td><td>20</td><td>Number of epochs for local training on each party</td></tr>
<tr><td>rounds</td><td>10</td><td>Number of rounds for global deferated training</td></tr>
<tr><td>model_dir Cell:1</td><td>'...'</td><td>Path of saved model</td></tr>
</table>
</div>

Then you can predict the results on the given dataset by the following command:
```
python predict.py --model_dir='M_NAME' –test_dir=’T_NAME’
```

# References
Idash: http://www.humangenomeprivacy.org/2021/competition-tasks.html

Tensorflow-federated: https://github.com/tensorflow/federated

Tensorflow-privacy: https://github.com/tensorflow/privacy

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpoint_management.py

https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/optimizers/dp_optimizer.py


