# Aivian
Bird detection model

This model uses datasets from kaggle.
To get the datasets, open your kaggle account and do the following:

1. Go to your account, Scroll to API section and Click Expire API Token to remove previous tokens

2. Click on Create New API Token - It will download kaggle.json file on your machine.

Copy the kaggle.json file in your working directory.
After this, open your terminal, shift to your working directory, and follow the following steps:
1. pip install -q kaggle
2. mkdir ~/.kaggle
3. cp kaggle.json ~/.kaggle
4. chmod 600 ~/.kaggle/kaggle.json
5. kaggle datasets download gpiosenka/100-bird-species
6. unzip 100-bird-species.zip

This will download the dataset to your directory. Now when you run main.py you can provide the required directories to train the model and replicate the results.
