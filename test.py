from train import *

import pandas as pd
from scipy import stats
import numpy as np
# from imblearn.over_sampling import SMOTE,SVMSMOTE

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,KFold,StratifiedKFold,train_test_split as split
from sklearn.utils import resample, shuffle
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,auc

from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

test = preprocess(test,nan_approach='mean',target_ordinal=False #target ordinal belom dicek di dataset baru
                   , drop_features=True)
test = scaler(test)

predict(test,model,'coba2_inikeren_proba')

