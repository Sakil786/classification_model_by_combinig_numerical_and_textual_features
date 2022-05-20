import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")


class ConstantValue:
    def __init__(self):
        self.file_path = r'D:\xgboos\data.xlsx'
        self.max_features_value = 5000
        self.test_size_value = 0.33
        self.random_state_value = 0
        self.one_value = 1
        self.three_value = 3

    def data_preparation(self, file):
        data = pd.read_excel(file)
        data = data.dropna()
        data_select = data[['retweet_count', 'followers_count', 'friends_count', 'processed_text', 'label']]
        data_select['all_features'] = data_select.retweet_count.astype(str) + data_select.followers_count.astype(
            str) + data_select.friends_count.astype(str) + data_select.processed_text
        # normal=0,hateful=1
        data_select['target'] = np.where(data_select['label'] == 'normal', 0, 1)
        data_select1 = data_select[['all_features', 'target']]
        print(data_select1.shape)
        return data_select1


obj = ConstantValue()
obj.data_preparation(obj.file_path)
