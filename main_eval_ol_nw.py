# -*- coding: utf-8 -*-

from os.path import join as join
import pandas as pd
import joblib

from modules.evaluate import evaluate_performance
from modules.print_draw import print_data
from models.models import model_predict

DATA_DIR = "data"
MODELS_DIR = "models"
TARGET_COL = "montant_pret"

# 1) Charger les datasets
df_old = pd.read_csv(join(DATA_DIR, "df_old.csv"))
df_new = pd.read_csv(join(DATA_DIR, "df_new.csv"))

# 2) Charger le preprocessseur et le modele
preprocessor = joblib.load(join(MODELS_DIR, "preprocessor.pkl"))
model = joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))

# 3) Fonction pour separer X et y puis transformer X
def make_X_y(df):
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    X_prep = preprocessor.transform(X_raw)
    return X_prep, y

# 4) Preparation des donnees
X_old, y_old = make_X_y(df_old)
X_new, y_new = make_X_y(df_new)

# 5) Predictions et performances sur df_old
y_pred_old = model_predict(model, X_old)
perf_old = evaluate_performance(y_old, y_pred_old)
print_data(perf_old, exp_name="Performance modele sur df_old")

# 6) Predictions et performances sur df_new
y_pred_new = model_predict(model, X_new)
perf_new = evaluate_performance(y_new, y_pred_new)
print_data(perf_new, exp_name="Performance modele sur df_new")
