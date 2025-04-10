import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

import PyIO
import PyPluMA

class PlotFIPlugin:
 def input(self, inputfile):
  self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  stat1_test = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["teststat"], sep="\t")
  origin_test = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["testorigin"], sep="\t")
  stats = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["stats"], sep="\t")
  model = XGBClassifier()
  model.load_model(PyPluMA.prefix()+"/"+self.parameters["model"])



  # Feature Importance
  feature_importance = model.feature_importances_features = stats.columns

  # Plot the top 7 features
  #xgboost.plot_importance(model, max_num_features=12)

  # Prediction Report
  y_pred = model.predict(stat1_test)
  report = classification_report(origin_test, y_pred)

  # Show the plot
  plt.show()
  plt.savefig(outputfile, dpi=1200, bbox_inches='tight')
  plt.close()

