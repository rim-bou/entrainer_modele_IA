# -*- coding: utf-8 -*-

from os.path import join as join
import pandas as pd
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt

from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data
from models.models import create_nn_model, train_model, model_predict

DATA_DIR = "data"


def save_loss_figure(history, filename):
    """
    Sauvegarde une figure de la courbe de loss a partir de l historique Keras.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Courbe de loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def train_and_eval_on_df(df, exp_label, epochs=80):
    """
    Entraine un modele sur un dataframe donne (df),
    loggue les infos dans MLflow et retourne le modele et les perfs.
    """

    # 1) Preprocessing: X, y
    X, y, _ = preprocessing(df)

    # 2) Split train / test
    X_train, X_test, y_train, y_test = split(X, y)

    # 3) Creation d un modele vierge
    input_dim = X_train.shape[1]
    model = create_nn_model(input_dim)

    # ----- MLflow : demarrage du run -----
    with mlflow.start_run(run_name=f"model_{exp_label}"):

        # Logger quelques parametres
        mlflow.log_param("dataset_label", exp_label)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("input_dim", int(input_dim))

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
            verbose=1,
        )

        # 5) Perf sur train
        y_pred_train = model_predict(model, X_train)
        perf_train = evaluate_performance(y_train, y_pred_train)
        print_data(perf_train, exp_name=f"{exp_label} - TRAIN")

        # Logger les metriques train
        mlflow.log_metric("train_MSE", perf_train["MSE"])
        mlflow.log_metric("train_MAE", perf_train["MAE"])
        mlflow.log_metric("train_R2", perf_train["R2"])

        # 6) Perf sur test
        y_pred_test = model_predict(model, X_test)
        perf_test = evaluate_performance(y_test, y_pred_test)
        print_data(perf_test, exp_name=f"{exp_label} - TEST")

        # Logger les metriques test
        mlflow.log_metric("test_MSE", perf_test["MSE"])
        mlflow.log_metric("test_MAE", perf_test["MAE"])
        mlflow.log_metric("test_R2", perf_test["R2"])

        # 7) Courbe de loss -> sauvegarde + artifact MLflow
        loss_fig_path = f"loss_{exp_label}.png"
        save_loss_figure(hist, loss_fig_path)
        mlflow.log_artifact(loss_fig_path)

        # 8) Logger le modele dans MLflow
        mlflow.keras.log_model(model, artifact_path="model")

    # Le run MLflow est termine ici
    return model, hist, perf_train, perf_test


def main():
    # Nom de l experience MLflow
    mlflow.set_experiment("credit_pret_compare_old_new")

    # Charger les deux jeux de donnees
    df_old = pd.read_csv(join(DATA_DIR, "df_old.csv"))
    df_new = pd.read_csv(join(DATA_DIR, "df_new.csv"))

    # Entrainement sur df_old -> 1 run MLflow
    model_old, hist_old, perf_train_old, perf_test_old = train_and_eval_on_df(
        df_old, exp_label="OLD", epochs=80
    )

    # Entrainement sur df_new -> 1 autre run MLflow
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

