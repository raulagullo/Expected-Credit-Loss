# =====================================================================
# OPTIMIZACIÓN DEL LightGBM DE PD (linear trees) CON VALIDACIÓN CRUZADA DENTRO DE TRAIN
# - Objetivo de Optuna: log-loss media en validación cruzada estratificada (3 bloques) sobre una submuestra de train.
# - Validación NO se usa para elegir nada: queda reservada para la calibración final de la ECL.
#   Aquí solo se usa al final para informar de las métricas del modelo elegido frente al actual.
# - Toda la aleatoriedad depende de SEED: submuestra, bloques de la CV, muestreador de Optuna, LightGBM
#   y el 10 % del early stopping final.
# - Early stopping siempre con datos de train (el bloque de validación interna de cada fold, o un 10 % de train
#   en el reentrenamiento final), nunca con el conjunto de validación.
# =====================================================================
import time
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

SEED = 42

try:                                   # en Jupyter/Colab existe display; ejecutado como script, se imprime
    from IPython.display import display
except ImportError:
    display = print
pd.set_option("display.width", 250, "display.max_columns", 30)

optuna.logging.set_verbosity(optuna.logging.WARNING)
assert "SEED" in globals(), "Define SEED antes de ejecutar esta celda"

import os
# Carpeta donde se guardan el estudio y los resultados y donde está pd_val_gbm_lt_1.parquet.
# En Colab, la del checkpoint en Drive; fuera de Colab, la carpeta desde la que se ejecuta el script.
RUTA_COLAB = "/content/drive/MyDrive/Proyectos Data Science/checkpoint_ead/"
RUTA = RUTA_COLAB if os.path.isdir(RUTA_COLAB) else os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), "")
RUTA = RUTA.replace("\\", "/")                        # barras normales: válidas en Windows y en la URL de SQLite
print(f"Carpeta de resultados: {RUTA}")

# Comprobaciones ANTES de las horas de búsqueda: todo lo que se necesita al final debe existir ya
for f in ["X_train_pred.parquet", "X_val_pred.parquet", "y_train.parquet", "y_val.parquet",
          "pd_val_gbm_lt_1.parquet"]:
    assert os.path.exists(f"{RUTA}{f}"), f"No encuentro {f} en {RUTA}"
# Base de datos del estudio: cada ensayo se guarda al terminar. Si el proceso se corta, al relanzar se continúa
# donde se quedó en lugar de repetir los ensayos ya hechos.
SUFIJO = "_oof"          # los resultados de la ejecución anterior (WoE dentro de muestra) no se sobrescriben
ALMACEN_OPTUNA = f"sqlite:///{RUTA}pd_optuna_cv{SUFIJO}.db"
NOMBRE_ESTUDIO = f"pd_gbm_lt_seed{SEED}"

# ===== Datos: las mismas matrices WoE que la tercera regresión logística y el modelo actual =====
# (se cargan con nombres propios para no pisar X_train / X_val del resto del notebook)
X_tr_pd = pd.read_parquet(f"{RUTA}X_train_pred.parquet")
X_va_pd = pd.read_parquet(f"{RUTA}X_val_pred.parquet")
y_tr_pd_s = pd.read_parquet(f"{RUTA}y_train.parquet").squeeze()
y_va_pd_s = pd.read_parquet(f"{RUTA}y_val.parquet").squeeze()
assert X_tr_pd.index.equals(y_tr_pd_s.index) and X_va_pd.index.equals(y_va_pd_s.index)
y_tr_pd = y_tr_pd_s.astype(int).to_numpy()
y_va_pd = y_va_pd_s.astype(int).to_numpy()
print(f"Train: {X_tr_pd.shape} | tasa de impago {y_tr_pd.mean():.4f}  ·  Val: {X_va_pd.shape} | tasa {y_va_pd.mean():.4f}")

FRAC_BUSQUEDA = 0.3      # fracción de train usada en la búsqueda (linear trees es lento con todo train)
N_FOLDS = 3
N_TRIALS = 40
TIMEOUT_MIN = None        # sin tope de tiempo: con timeout, el nº de ensayos dependería de la velocidad
                          # de la máquina y el resultado dejaría de ser reproducible con la misma SEED
ES_RONDAS = 100           # early stopping: rondas sin mejora

def error_calibracion(y, p, n_grupos=10):
    """Media, ponderada por tamaño, de |PD media − tasa de impago| en deciles de la PD predicha."""
    g = pd.qcut(pd.Series(p).rank(method="first"), n_grupos, labels=False)
    t = pd.DataFrame({"g": g.to_numpy(), "y": y, "p": p}).groupby("g").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((np.abs(t["p"] - t["y"]) * t["n"]).sum() / t["n"].sum())

def metricas_pd(y, p):
    return {"log_loss": log_loss(y, p), "AUC": roc_auc_score(y, p), "Gini": 2 * roc_auc_score(y, p) - 1,
            "Brier": brier_score_loss(y, p), "error_calibración": error_calibracion(y, p),
            "PD media": float(np.mean(p)), "tasa impago": float(np.mean(y))}

# --- Submuestra estratificada de train para la búsqueda ---
idx_busq, _ = train_test_split(np.arange(len(y_tr_pd)), train_size=FRAC_BUSQUEDA,
                               stratify=y_tr_pd, random_state=SEED)
X_b, y_b = X_tr_pd.iloc[idx_busq], y_tr_pd[idx_busq]
FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_b, y_b))

def params_base():
    # random_state=SEED fija todas las semillas internas de LightGBM (bagging, columnas, etc.);
    # deterministic + force_row_wise hacen que el resultado no dependa del orden de los hilos
    return dict(objective="binary", linear_tree=True, n_estimators=4000,
                subsample_freq=1, random_state=SEED, deterministic=True, force_row_wise=True,
                verbose=-1, n_jobs=-1)

def objetivo(trial):
    params = {**params_base(),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
              "min_child_samples": trial.suggest_int("min_child_samples", 200, 5000, log=True),
              "linear_lambda": trial.suggest_float("linear_lambda", 0.01, 100, log=True),
              "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
              "subsample": trial.suggest_float("subsample", 0.4, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0)}
    perdidas, iteraciones = [], []
    for k, (i_fit, i_es) in enumerate(FOLDS):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_b.iloc[i_fit], y_b[i_fit], eval_set=[(X_b.iloc[i_es], y_b[i_es])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(ES_RONDAS, verbose=False)])
        p = m.predict_proba(X_b.iloc[i_es], num_iteration=m.best_iteration_)[:, 1]
        perdidas.append(log_loss(y_b[i_es], p))
        iteraciones.append(m.best_iteration_)
        trial.report(float(np.mean(perdidas)), k)          # poda: se abandona si va peor que la mediana
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("iteraciones_medias", float(np.mean(iteraciones)))
    return float(np.mean(perdidas))

estudio = optuna.create_study(direction="minimize", storage=ALMACEN_OPTUNA, study_name=NOMBRE_ESTUDIO,
                              load_if_exists=True,
                              sampler=optuna.samplers.TPESampler(seed=SEED),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
# Primeros ensayos de referencia en la misma CV (solo si el estudio no tiene ninguno terminado o pendiente):
#   0 = modelo inicial (gbm_lt_1); 1 = mejor ensayo (38) de la búsqueda anterior con el WoE dentro de muestra
if not any(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                       optuna.trial.TrialState.WAITING) for t in estudio.trials):
    estudio.enqueue_trial({"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 1000, "linear_lambda": 1.0,
                           "reg_lambda": 1e-3, "subsample": 0.5, "colsample_bytree": 0.8})
    estudio.enqueue_trial({"learning_rate": 0.010361, "num_leaves": 57, "min_child_samples": 363,
                           "linear_lambda": 36.890709, "reg_lambda": 0.001245, "subsample": 0.967926,
                           "colsample_bytree": 0.539114})
# Ensayos que faltan: los terminados o podados ya cuentan (los que quedaron a medias por un corte no)
hechos = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in estudio.trials)
t0 = time.time()
if hechos < N_TRIALS:
    estudio.optimize(objetivo, n_trials=N_TRIALS - hechos,
                     timeout=None if TIMEOUT_MIN is None else TIMEOUT_MIN * 60,
                     n_jobs=1, show_progress_bar=True)   # n_jobs=1: ensayos en serie, en orden reproducible
print(f"Búsqueda: {len(estudio.trials)} ensayos ({hechos} ya guardados) · {(time.time() - t0) / 60:.0f} min en esta ejecución")

ensayos = estudio.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
ensayos = ensayos[ensayos["state"] == "COMPLETE"].sort_values("value")
display(ensayos.head(10))
ensayos.to_csv(f"{RUTA}pd_optuna_ensayos{SUFIJO}.csv", index=False)
ref0, ref38 = estudio.trials[0].value, estudio.trials[1].value
print(f"Log-loss CV · modelo inicial: {ref0:.5f} | ensayo 38 anterior: {ref38:.5f} | mejor: {estudio.best_value:.5f} "
      f"(ensayo {estudio.best_trial.number})")

# ===== Reentrenamiento final con todo train (early stopping con un 10 % interno de train) =====
BEST_PARAMS_PD = {**params_base(), **estudio.best_params}
i_fit, i_es = train_test_split(np.arange(len(y_tr_pd)), test_size=0.10, stratify=y_tr_pd,
                               random_state=SEED)                 # 10 % de train para el early stopping
model_gbm_lt_2 = lgb.LGBMClassifier(**BEST_PARAMS_PD)
model_gbm_lt_2.fit(X_tr_pd.iloc[i_fit], y_tr_pd[i_fit], eval_set=[(X_tr_pd.iloc[i_es], y_tr_pd[i_es])],
                   eval_metric="binary_logloss", callbacks=[lgb.early_stopping(ES_RONDAS, verbose=False)])
N_ARBOLES_PD = model_gbm_lt_2.best_iteration_
print(f"Árboles del modelo final: {N_ARBOLES_PD}")

pd_val_gbm_lt_2 = model_gbm_lt_2.predict_proba(X_va_pd, num_iteration=N_ARBOLES_PD)[:, 1]

# Se guardan el modelo y su PD en validación (ejecutado como script, las variables se pierden al terminar)
model_gbm_lt_2.booster_.save_model(f"{RUTA}model_gbm_lt_optuna{SUFIJO}.txt", num_iteration=N_ARBOLES_PD)
pd.DataFrame({"pd": pd_val_gbm_lt_2}, index=X_va_pd.index).to_parquet(f"{RUTA}pd_val_gbm_lt_optuna{SUFIJO}.parquet")

# ===== Comparación en validación (solo informativa: la elección ya está hecha con la CV de train) =====
comparacion = {"optimizado (gbm_lt_2)": metricas_pd(y_va_pd, pd_val_gbm_lt_2)}
# PD del modelo actual: de la sesión si existe; si no, del parquet guardado en el checkpoint
pd_actual = pd.read_parquet(f"{RUTA}pd_val_gbm_lt_1.parquet").squeeze()
pd_actual = pd.Series(np.asarray(pd_actual).ravel(), index=getattr(pd_actual, "index", None))
pd_actual = pd_actual.loc[X_va_pd.index] if X_va_pd.index.isin(pd_actual.index).all() else pd_actual
assert len(pd_actual) == len(y_va_pd), "La PD del modelo actual no cuadra con la validación"
comparacion = {"actual (gbm_lt_1)": metricas_pd(y_va_pd, pd_actual.to_numpy()), **comparacion}
tabla_comparacion = pd.DataFrame(comparacion).T.round(5)     # sin .style: así también se ve bien con print
display(tabla_comparacion)
tabla_comparacion.to_csv(f"{RUTA}pd_optuna_comparacion_val{SUFIJO}.csv")