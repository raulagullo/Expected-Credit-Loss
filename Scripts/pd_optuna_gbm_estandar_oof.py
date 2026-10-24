# =====================================================================
# OPTIMIZACIÓN DEL LightGBM ESTÁNDAR DE PD (linear_tree=False) CON VALIDACIÓN CRUZADA DENTRO DE TRAIN
# - Mismas matrices que el LightGBM linear trees: X_train_pred / X_val_pred con el WoE fuera de muestra.
# - Objetivo de Optuna: log-loss media en validación cruzada estratificada (3 bloques) sobre una muestra de train.
# - Validación NO se usa para elegir nada; solo al final para comparar con el LightGBM linear trees optimizado.
# - Toda la aleatoriedad depende de SEED.
# - Early stopping siempre con datos de train (el bloque reservado de cada fold, o un 10 % de train
#   en el reentrenamiento final).
# - Cada ensayo se guarda en SQLite: si el proceso se corta, al relanzarlo continúa donde se quedó.
#
# Ficheros necesarios en la carpeta del script:
#   X_train_pred.parquet, X_val_pred.parquet, y_train.parquet, y_val.parquet,
#   pd_val_gbm_lt_optuna_oof.parquet (PD en validación del LightGBM linear trees optimizado)
# =====================================================================
import os
import time
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

try:                                   # en Jupyter existe display; ejecutado como script, se imprime
    from IPython.display import display
except ImportError:
    display = print
pd.set_option("display.width", 250, "display.max_columns", 30)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------- Configuración -----------------------------
SEED = 42
RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), "")
RUTA = RUTA.replace("\\", "/")

FRAC_BUSQUEDA = 0.3      # fracción de train usada en la búsqueda
N_FOLDS = 3
N_TRIALS = 40
ES_RONDAS = 100          # early stopping: rondas sin mejora

SUFIJO = "_estandar_oof"
ALMACEN_OPTUNA = f"sqlite:///{RUTA}pd_optuna_cv{SUFIJO}.db"
NOMBRE_ESTUDIO = f"pd_gbm_estandar_seed{SEED}"

# ----------------------------- Datos -----------------------------
for f in ["X_train_pred.parquet", "X_val_pred.parquet", "y_train.parquet", "y_val.parquet",
          "pd_val_gbm_lt_optuna_oof.parquet"]:
    assert os.path.exists(f"{RUTA}{f}"), f"No encuentro {f} en {RUTA}"

X_tr = pd.read_parquet(f"{RUTA}X_train_pred.parquet")
X_va = pd.read_parquet(f"{RUTA}X_val_pred.parquet")
y_tr_s = pd.read_parquet(f"{RUTA}y_train.parquet").squeeze()
y_va_s = pd.read_parquet(f"{RUTA}y_val.parquet").squeeze()
assert X_tr.index.equals(y_tr_s.index) and X_va.index.equals(y_va_s.index), "Índices de X e y no alineados"
y_tr, y_va = y_tr_s.astype(int).to_numpy(), y_va_s.astype(int).to_numpy()
print(f"Carpeta: {RUTA}")
print(f"Train: {X_tr.shape} | tasa {y_tr.mean():.4f}  ·  Val: {X_va.shape} | tasa {y_va.mean():.4f}")

# ----------------------------- Métricas -----------------------------
def error_calibracion(y, p, n_grupos=10):
    """Media, ponderada por tamaño, de |PD media − tasa de impago| en deciles de la PD predicha."""
    g = pd.qcut(pd.Series(p).rank(method="first"), n_grupos, labels=False)
    t = pd.DataFrame({"g": g.to_numpy(), "y": y, "p": p}).groupby("g").agg(y=("y", "mean"), p=("p", "mean"),
                                                                         n=("y", "size"))
    return float((np.abs(t["p"] - t["y"]) * t["n"]).sum() / t["n"].sum())

def metricas_pd(y, p):
    auc = roc_auc_score(y, p)
    return {"log_loss": log_loss(y, p), "AUC": auc, "Gini": 2 * auc - 1, "Brier": brier_score_loss(y, p),
            "error_calibración": error_calibracion(y, p), "PD media": float(np.mean(p)),
            "tasa impago": float(np.mean(y))}

# ----------------------------- Búsqueda -----------------------------
idx_b, _ = train_test_split(np.arange(len(y_tr)), train_size=FRAC_BUSQUEDA, stratify=y_tr, random_state=SEED)
X_b, y_b = X_tr.iloc[idx_b], y_tr[idx_b]
FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_b, y_b))

# deterministic + force_row_wise: el resultado no depende del orden de los hilos
PARAMS_BASE = dict(objective="binary", linear_tree=False, n_estimators=5000, subsample_freq=1,
                   random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1)

def objetivo(trial):
    params = {**PARAMS_BASE,
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),     # sin regresión en las hojas,
              "min_child_samples": trial.suggest_int("min_child_samples", 200, 5000, log=True),  # más hojas
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
    trial.set_user_attr("arboles_medios", float(np.mean(iteraciones)))
    return float(np.mean(perdidas))

estudio = optuna.create_study(direction="minimize", storage=ALMACEN_OPTUNA, study_name=NOMBRE_ESTUDIO,
                              load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
if not any(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                       optuna.trial.TrialState.WAITING) for t in estudio.trials):
    # Primer ensayo de referencia: configuración genérica de LightGBM (31 hojas, valores habituales)
    estudio.enqueue_trial({"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 1000,
                           "reg_lambda": 1e-3, "subsample": 0.8, "colsample_bytree": 0.8})
hechos = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in estudio.trials)
t0 = time.time()
if hechos < N_TRIALS:
    estudio.optimize(objetivo, n_trials=N_TRIALS - hechos, n_jobs=1, show_progress_bar=True)
print(f"Búsqueda: {len(estudio.trials)} ensayos ({hechos} ya guardados) · {(time.time() - t0) / 60:.0f} min")

ensayos = estudio.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
ensayos = ensayos[ensayos["state"] == "COMPLETE"].sort_values("value")
display(ensayos.head(10))
ensayos.to_csv(f"{RUTA}pd_optuna_ensayos{SUFIJO}.csv", index=False)
print(f"Log-loss CV · referencia: {estudio.trials[0].value:.5f} | mejor: {estudio.best_value:.5f} "
      f"(ensayo {estudio.best_trial.number})")
print(f"BEST_PARAMS_GBM_ESTANDAR = {estudio.best_params}")

# ----------------------------- Reentrenamiento final con todo train -----------------------------
i_fit, i_es = train_test_split(np.arange(len(y_tr)), test_size=0.10, stratify=y_tr, random_state=SEED)
t0 = time.time()
modelo = lgb.LGBMClassifier(**PARAMS_BASE, **estudio.best_params)
modelo.fit(X_tr.iloc[i_fit], y_tr[i_fit], eval_set=[(X_tr.iloc[i_es], y_tr[i_es])], eval_metric="binary_logloss",
           callbacks=[lgb.early_stopping(ES_RONDAS, verbose=False)])
N_ARBOLES = modelo.best_iteration_
print(f"Reentrenamiento final: {(time.time() - t0) / 60:.1f} min | árboles: {N_ARBOLES}")

pd_val_estandar = modelo.predict_proba(X_va, num_iteration=N_ARBOLES)[:, 1]
modelo.booster_.save_model(f"{RUTA}model_gbm_optuna{SUFIJO}.txt", num_iteration=N_ARBOLES)
pd.DataFrame({"pd": pd_val_estandar}, index=X_va.index).to_parquet(f"{RUTA}pd_val_gbm_optuna{SUFIJO}.parquet")

# ----------------------------- Comparación en validación con el LightGBM linear trees -----------------------------
pd_val_lt = pd.read_parquet(f"{RUTA}pd_val_gbm_lt_optuna_oof.parquet")["pd"].loc[X_va.index].to_numpy()
comparacion = pd.DataFrame({
    "LightGBM linear trees optimizado": metricas_pd(y_va, pd_val_lt),
    "LightGBM estándar optimizado": metricas_pd(y_va, pd_val_estandar),
}).T.round(5)
print("\nComparación en validación:")
display(comparacion)
comparacion.to_csv(f"{RUTA}pd_optuna_comparacion_val{SUFIJO}.csv")