# =====================================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS DEL LINEAR TREE (LinearTreeClassifier) PARA PD CON OPTUNA
# - Mismas matrices que la tercera regresión logística: X_train_pred / X_val_pred (numéricas estandarizadas
#   + buro_sin_registro + WoE de las 8 categóricas).
# - Objetivo de Optuna: log-loss media en validación cruzada estratificada sobre una muestra de train.
#   Validación NO se usa para elegir nada; solo para comparar al final el modelo elegido con la regresión logística.
# - Toda la aleatoriedad depende de SEED (muestra, bloques, muestreador de Optuna).
# - Cada ensayo se guarda en SQLite: si el proceso se corta, al relanzarlo continúa donde se quedó.
#
# Ficheros necesarios en RUTA (exportados desde el notebook de desarrollo):
#   X_train_pred.parquet, X_val_pred.parquet, y_train.parquet, y_val.parquet
# =====================================================================
import os
os.environ["PYTHONWARNINGS"] = "ignore"     # los procesos paralelos (n_jobs=-1) heredan esto y tampoco muestran avisos
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
from lineartree import LinearTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")            # avisos de convergencia de las logísticas de cada nodo
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:                                         # en Jupyter existe display; ejecutado como script, se imprime
    from IPython.display import display
except ImportError:
    display = print
pd.set_option("display.width", 250, "display.max_columns", 30)

# ----------------------------- Configuración -----------------------------
SEED = 42                                    # la misma del notebook de desarrollo
RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), "")
RUTA = RUTA.replace("\\", "/")

N_BUSQUEDA = 200_000      # préstamos de train usados en la búsqueda (el linear tree es lento)
N_FOLDS = 3
N_TRIALS = 30
N_FINAL = None            # préstamos para el reentrenamiento final (None = todo train). Bájalo si tarda demasiado.

ALMACEN_OPTUNA = f"sqlite:///{RUTA}lt_optuna_cv.db"
NOMBRE_ESTUDIO = f"pd_linear_tree_seed{SEED}"

# ----------------------------- Datos -----------------------------
for f in ["X_train_pred.parquet", "X_val_pred.parquet", "y_train.parquet", "y_val.parquet"]:
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
            "error_calibración": error_calibracion(y, p), "PD media": float(np.mean(p)), "tasa impago": float(np.mean(y))}

# ----------------------------- Modelo -----------------------------
def crear_linear_tree(params):
    return LinearTreeClassifier(
        base_estimator=LogisticRegression(C=params["C"], max_iter=1000),
        criterion="crossentropy",
        max_depth=params["max_depth"],
        min_samples_split=6,
        min_samples_leaf=params["min_samples_leaf"],
        max_bins=params["max_bins"],
        min_impurity_decrease=params["min_impurity_decrease"],
        n_jobs=-1,
    )

# Muestra estratificada de train para la búsqueda y bloques de la validación cruzada
idx_busq, _ = train_test_split(np.arange(len(y_tr)), train_size=min(N_BUSQUEDA, len(y_tr)) / len(y_tr),
                               stratify=y_tr, random_state=SEED)
X_b, y_b = X_tr.iloc[idx_busq], y_tr[idx_busq]
FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_b, y_b))

def objetivo(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.01, 0.20, log=True),   # fracción de la muestra
        "max_bins": trial.suggest_int("max_bins", 10, 40, step=5),
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),        # inversa de la regularización de cada logística
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-6, 1e-2, log=True),  # mejora mínima de log-loss por división
    }
    perdidas, hojas = [], []
    for k, (i_fit, i_ev) in enumerate(FOLDS):
        m = crear_linear_tree(params).fit(X_b.iloc[i_fit], y_b[i_fit])
        p = np.clip(m.predict_proba(X_b.iloc[i_ev])[:, 1], 1e-6, 1 - 1e-6)
        perdidas.append(log_loss(y_b[i_ev], p))
        hojas.append(len(m.summary(only_leaves=True)))
        trial.report(float(np.mean(perdidas)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("hojas_medias", float(np.mean(hojas)))
    return float(np.mean(perdidas))

# ----------------------------- Búsqueda -----------------------------
estudio = optuna.create_study(direction="minimize", storage=ALMACEN_OPTUNA, study_name=NOMBRE_ESTUDIO,
                              load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
if not any(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                       optuna.trial.TrialState.WAITING) for t in estudio.trials):
    # Primer ensayo: la configuración actual del notebook (C = 1 es el valor por defecto de LogisticRegression)
    estudio.enqueue_trial({"max_depth": 6, "min_samples_leaf": 0.1, "max_bins": 20, "C": 1.0,
                           "min_impurity_decrease": 1e-6})
hechos = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in estudio.trials)
t0 = time.time()
if hechos < N_TRIALS:
    estudio.optimize(objetivo, n_trials=N_TRIALS - hechos, n_jobs=1, show_progress_bar=True)
print(f"Búsqueda: {len(estudio.trials)} ensayos ({hechos} ya guardados) · {(time.time() - t0) / 60:.0f} min")

ensayos = estudio.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
ensayos = ensayos[ensayos["state"] == "COMPLETE"].sort_values("value")
display(ensayos.head(10))
ensayos.to_csv(f"{RUTA}lt_optuna_ensayos.csv", index=False)
print(f"Log-loss CV · configuración actual: {estudio.trials[0].value:.5f} | mejor: {estudio.best_value:.5f}")
print(f"Mejores hiperparámetros: {estudio.best_params}")

# ----------------------------- Reentrenamiento final y comparación en validación -----------------------------
if N_FINAL is None or N_FINAL >= len(y_tr):
    i_final = np.arange(len(y_tr))
else:
    i_final, _ = train_test_split(np.arange(len(y_tr)), train_size=N_FINAL / len(y_tr),
                                  stratify=y_tr, random_state=SEED)
t0 = time.time()
modelo_lt = crear_linear_tree(estudio.best_params).fit(X_tr.iloc[i_final], y_tr[i_final])
print(f"Reentrenamiento final con {len(i_final):,} préstamos: {(time.time() - t0) / 60:.1f} min | "
      f"hojas: {len(modelo_lt.summary(only_leaves=True))}")

pd_val_lt = np.clip(modelo_lt.predict_proba(X_va)[:, 1], 1e-6, 1 - 1e-6)
joblib.dump(modelo_lt, f"{RUTA}modelo_lt_optuna.joblib")
pd.DataFrame({"pd": pd_val_lt}, index=X_va.index).to_parquet(f"{RUTA}pd_val_lt_optuna.parquet")

# Regresión logística de referencia (tercera regresión del notebook, misma matriz)
logistica = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
comparacion = pd.DataFrame({
    "regresión logística": metricas_pd(y_va, logistica.predict_proba(X_va)[:, 1]),
    "linear tree optimizado": metricas_pd(y_va, pd_val_lt),
}).T.round(5)
print("\nComparación en validación:")
display(comparacion)
comparacion.to_csv(f"{RUTA}lt_optuna_comparacion_val.csv")