# -*- coding: utf-8 -*-

from os.path import join as join
import pandas as pd

from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict

DATA_DIR = "data"


def train_and_eval_on_df(df, exp_label, epochs=80):
    """
    Entraine un nouveau modele sur un dataframe donne (df)
    puis affiche les performances et la courbe de loss.

    exp_label : texte pour identifier l experience (par ex. "OLD" ou "NEW")
    """

    # 1) Preprocessing: X, y
    X, y, _ = preprocessing(df)

    # 2) Split train / test
    X_train, X_test, y_train, y_test = split(X, y)

    # 3) Creation d un modele vierge
    input_dim = X_train.shape[1]
    model = create_nn_model(input_dim)

    # 4) Entrainement
    print(f"\n========== Entrainement modele sur {exp_label} ==========")
    model, hist = train_model(
        model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    # 5) Perf sur train
    y_pred_train = model_predict(model, X_train)
    perf_train = evaluate_performance(y_train, y_pred_train)
    print_data(perf_train, exp_name=f"{exp_label} - TRAIN")

    # 6) Perf sur test
    y_pred_test = model_predict(model, X_test)
    perf_test = evaluate_performance(y_test, y_pred_test)
    print_data(perf_test, exp_name=f"{exp_label} - TEST")

    # 7) Courbe de loss
    print(f"Affichage de la courbe de loss pour {exp_label}")
    draw_loss(hist)

    return model, hist, perf_train, perf_test


def main():
    # Charger les deux jeux de donnees
    df_old = pd.read_csv(join(DATA_DIR, "df_old.csv"))
    df_new = pd.read_csv(join(DATA_DIR, "df_new.csv"))

    # Entrainement sur df_old
    model_old, hist_old, perf_train_old, perf_test_old = train_and_eval_on_df(
        df_old, exp_label="OLD", epochs=80
    )

    # Entrainement sur df_new
    model_new, hist_new, perf_train_new, perf_test_new = train_and_eval_on_df(
        df_new, exp_label="NEW", epochs=80
    )

    print("\n========== RESUME COMPARAISON ==========")
    print("Modele OLD (entraine sur df_old) - TEST :")
    print(perf_test_old)
    print("\nModele NEW (entraine sur df_new) - TEST :")
    print(perf_test_new)
    print("=========================================")


if __name__ == "__main__":
    main()
