# =====================================================================
# PIPELINE COMPLETO DEL RETO ECL COMO FUNCIÓN DE LA SEMILLA + BUCLE DE SEMILLAS
# Basado en el notebook "Riesgo_Cartera_Pipeline_Final.ipynb" (sin el apartado de métricas).
#
# Uso en PyCharm:
#   - X.csv, target.csv y emp_title_clasification.py en la carpeta de RUTA_DATOS
#     (por defecto, la misma carpeta que este script).
#   - Elige el modelo de PD en MODELO_PD (sección 2). Los resultados se guardan después de cada semilla en
#     un CSV propio de ese modelo, resultados_semillas_<MODELO_PD>.csv, así que las pruebas con distintos
#     hiperparámetros de PD no se mezclan. Si el proceso se corta, al relanzarlo se saltan las semillas ya hechas.
#
# Importante: el bucle es solo una COMPROBACIÓN DE ROBUSTEZ. Los tests de otras semillas se solapan con
# el test oficial, así que sus resultados no deben usarse para cambiar parámetros del modelo.
# =====================================================================
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")                       # avisos de LightGBM / statsmodels esperados
pd.set_option("display.width", 250, "display.max_columns", 30)

# ----------------------------- Rutas -----------------------------
RUTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = r"C:\Users\raula\Desktop\Data Science\Outliers In Data\Riesgo_Credito_Reto_Septiembre"                               # carpeta con X.csv, target.csv y emp_title_clasification.py
sys.path.insert(0, RUTA_DATOS)
from emp_title_clasification import ClasificadorEmpTitle   # noqa: E402

# ----------------------------- Semillas a evaluar -----------------------------
SEMILLAS = [42, #semilla notebook desarrollo
            89723, 89, 1, 7, 13, 21, 111111, 123, 2026, 31415,      # primer lote
            4821, 59, 3307, 76543, 248, 9001, 1618, 50505, 734, 27182]  # segundo lote

# ----------------------------- Configuración del split -----------------------------
PERC_TEST = 0.20
MIN_TRAIN = 0.60
PERC_VAL = min(PERC_TEST, 1 - MIN_TRAIN - PERC_TEST)
assert PERC_VAL > 0, "PERC_TEST demasiado grande: no queda sitio para validación con train ≥ MIN_TRAIN"
STRATIFY = False

# =====================================================================
# 1a. INGENIERÍA DE VARIABLES COMÚN
# =====================================================================
VARIABLES_DESEMPEÑO = ["remaining_princ_for_tot_amnt_fund", "paym_rec", "princ_rec", "interest_rec", "late_fees_rec"]

MAPA_EMP_LENGTH = {
    "< 1 year": "< 2 years", "1 year": "< 2 years", "2 years": "2 years", "3 years": "3 years",
    "4 years": "4 years", "5 years": "5-6 years", "6 years": "5-6 years", "7 years": "7 years",
    "8 years": "8 years", "9 years": "9 years", "10+ years": "10+ years", "unknown": "unknown",
}
MESES = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
         "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
VENTANA_BURO = 84
COLS_DELINQ = ["mths_since_last_delinq", "mths_since_recent_revol_delinq", "mths_since_recent_bankcard_delinq"]
BLOQUE_BURO_COMPLETO = ["max_bal_owed", "num_rev_trades_op_in_12mths", "num_inq", "num_installment_acc_op_in_12mths",
                        "num_open_trades_in_6mths", "num_installment_acc_op_in_24mths", "num_rev_trades_op_in_24mths"]
CORTES_INSTALL = [0, 3, 6, 9, 12, 15, 18, 21, 30, 60, np.inf]
ETIQUETAS_INSTALL = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-30", "30-60", ">60"]
CORTES_BAL = [0, 25, 50, 75, 100, np.inf]
ETIQUETAS_BAL = ["0-25", "25-50", "50-75", "75-100", ">100"]
CORTES_DEROG = [-0.1, 0, 1, 2, 5, np.inf]
ETIQUETAS_DEROG = ["0", "1", "2", "3-5", ">=6"]
CORTES_LAST_DELINQ = [0, 6, 18, 36, 60, 72, 84]
ETIQUETAS_LAST_DELINQ = ["0-6", "6-18", "18-36", "36-60", "60-72", "72-84"]
COLUMNAS_ELIMINADAS = [
    "emp_length", "emp_title", "emp_title_norm", "emp_tipo", "region_code",
    "mths_since_last_installment_acc_op", "bal_to_cred_lim", "annual_income", "num_inq_in_12mths",
    "monthly_payment", "earliest_cr_line_year", "earliest_cr_line_month",
    "mths_since_recent_revol_delinq", "mths_since_recent_bankcard_delinq",
    "never_revol_delinq", "never_bankcard_delinq",
    "num_installment_acc_op_in_12mths", "num_rev_trades_op_in_12mths",
]


def split_train_val_test(X, y, perc_val, perc_test, seed, stratify=True):
    """Divide en train / val / test. El test depende solo de seed y perc_test."""
    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=perc_test, random_state=seed, stratify=y if stratify else None)
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto, test_size=perc_val / (1 - perc_test), random_state=seed,
        stratify=y_resto if stratify else None)
    return (X_train.copy(), X_val.copy(), X_test.copy(), y_train.copy(), y_val.copy(), y_test.copy())


def preprocesar(X_train, X_val, X_test):
    """Ingeniería de variables común a los tres modelos. No modifica las tablas de entrada."""
    conjuntos = {"train": X_train.copy(), "val": X_val.copy(), "test": X_test.copy()}

    tr = conjuntos["train"]
    emp_tr = tr["emp_title"].str.strip()
    emp_tr = emp_tr.mask(emp_tr.str.lower().isin({"", "na", "test", "."})).fillna("unknown")
    clasificador_emp = ClasificadorEmpTitle().fit(emp_tr)
    mediana_bal = tr["bal_to_cred_lim"].median()

    for df in conjuntos.values():
        df.rename(columns={"paym_rec_for_tot_amnt_fund": "paym_rec"}, inplace=True)

        emp = df["emp_title"].str.strip()
        df["emp_title"] = emp.mask(emp.str.lower().isin({"", "na", "test", "."})).fillna("unknown")
        df["emp_length_cat"] = df["emp_length"].fillna("unknown").map(MAPA_EMP_LENGTH)
        df[["emp_title_norm", "emp_tipo", "emp_ambito"]] = clasificador_emp.transform(df["emp_title"])

        df["credit_history_months"] = (
            (pd.to_numeric(df["issue_date_year"], errors="coerce")
             - pd.to_numeric(df["earliest_cr_line_year"], errors="coerce")) * 12
            + (df["issue_date_month"].str.strip().map(MESES) - df["earliest_cr_line_month"].str.strip().map(MESES)))

        df[COLS_DELINQ] = df[COLS_DELINQ].clip(upper=VENTANA_BURO)
        df["never_delinq"] = df["mths_since_last_delinq"].isna().astype(int)
        periodo_verificable = df["credit_history_months"].clip(lower=0, upper=VENTANA_BURO)
        for c in COLS_DELINQ:
            df[c] = df[c].fillna(periodo_verificable)

        bloque_nulo = df[BLOQUE_BURO_COMPLETO].isna().all(axis=1)
        v = df["mths_since_last_installment_acc_op"]
        cat = pd.cut(v, bins=CORTES_INSTALL, labels=ETIQUETAS_INSTALL, include_lowest=True).astype("object")
        cat[v.isna() & bloque_nulo] = "sin_registro"
        cat[v.isna() & ~bloque_nulo] = "nunca"
        df["mths_since_last_installment_acc_op_cat"] = cat

        extra = df["bal_to_cred_lim"].isna() & ~bloque_nulo
        df.loc[extra, "bal_to_cred_lim"] = mediana_bal
        cat = pd.cut(df["bal_to_cred_lim"], bins=CORTES_BAL, labels=ETIQUETAS_BAL,
                     include_lowest=True).astype("object")
        cat[bloque_nulo] = "sin_registro"
        df["bal_to_cred_lim_cat"] = cat

        df["annual_income_log"] = np.log10(df["annual_income"])
        df["num_inq_in_7_12mths"] = (df["num_inq_in_12mths"] - df["num_inq_in_6mths"]).clip(lower=0)
        df["num_derogatory_pub_rec_cat"] = pd.cut(df["num_derogatory_pub_rec"], bins=CORTES_DEROG,
                                                  labels=ETIQUETAS_DEROG).astype("object")
        cat = pd.cut(df["mths_since_last_delinq"], bins=CORTES_LAST_DELINQ, labels=ETIQUETAS_LAST_DELINQ,
                     include_lowest=True).astype("object")
        cat[df["never_delinq"] == 1] = "sin retraso registrado"
        df["mths_since_last_delinq_cat"] = cat

        df.drop(columns=[c for c in COLUMNAS_ELIMINADAS if c in df.columns], inplace=True)

    for nombre, df in conjuntos.items():
        assert list(df.columns) == list(conjuntos["train"].columns), f"{nombre}: columnas distintas a train"
    return conjuntos["train"], conjuntos["val"], conjuntos["test"]


# =====================================================================
# 1b. MATRICES DE CADA MODELO
# =====================================================================
NUM_PD = ["interest_rate", "issue_date_year", "num_rev_trades_op_in_24mths", "num_open_trades_in_6mths",
          "num_installment_acc_op_in_24mths", "max_bal_owed", "num_inq", "loan_term_months",
          "dept_paym_income_ratio", "used_credit_share", "num_inq_in_6mths", "annual_income_log", "funded_amnt"]
CAT_PD = ["bal_to_cred_lim_cat", "mths_since_last_installment_acc_op_cat", "verification_status",
          "disbursement_method", "initial_list_status", "loan_purpose", "emp_ambito", "home_ownership_status"]
BLOQUE_BURO = ["num_open_trades_in_6mths", "num_installment_acc_op_in_24mths",
               "num_rev_trades_op_in_24mths", "max_bal_owed", "num_inq"]
MIN_N_WOE = 500
SUAVIZADO_WOE = 0.5
WOE_FUERA_DE_MUESTRA = True          # como en el notebook final
N_BLOQUES_WOE = 5


def tabla_woe(x, y):
    t = x.astype(str)
    frec = t.value_counts()
    t = t.where(t.isin(frec[frec >= MIN_N_WOE].index), "otros").where(x.notna(), "nulo")
    c = pd.crosstab(t, y)
    malos, buenos = c[1] + SUAVIZADO_WOE, c[0] + SUAVIZADO_WOE
    return np.log((buenos / buenos.sum()) / (malos / malos.sum())).rename("woe")


def aplicar_woe(x, woe):
    v = x.astype(str).where(x.notna(), "nulo")
    return v.where(v.isin(woe.index), "otros").map(woe).fillna(0).astype("float32")


def woe_train_fuera_de_muestra(X_train, y_train, seed):
    woe_tr = pd.DataFrame(index=X_train.index, columns=[f"{c}_woe" for c in CAT_PD], dtype="float32")
    for i_aj, i_ap in KFold(N_BLOQUES_WOE, shuffle=True, random_state=seed).split(X_train):
        i_aj, i_ap = X_train.index[i_aj], X_train.index[i_ap]
        for c in CAT_PD:
            woe = tabla_woe(X_train.loc[i_aj, c], y_train.loc[i_aj])
            woe_tr.loc[i_ap, f"{c}_woe"] = aplicar_woe(X_train.loc[i_ap, c], woe).to_numpy()
    return woe_tr.astype("float32")


def matrices_pd(X_train, X_val, X_test, y_train, seed):
    woes = {c: tabla_woe(X_train[c], y_train) for c in CAT_PD}
    bloque_pd = [c for c in BLOQUE_BURO if c in NUM_PD]

    def numericas(df):
        d = df[NUM_PD].copy()
        d["buro_sin_registro"] = df[BLOQUE_BURO].isna().all(axis=1).astype(int)
        return d

    num_tr = numericas(X_train)
    medianas = num_tr[bloque_pd].median()
    num_tr = num_tr.fillna(medianas)
    medias, desv = num_tr.mean(), num_tr.std()

    def preparar(df, woe_df=None):
        num = ((numericas(df).fillna(medianas) - medias) / desv).astype("float32")
        if woe_df is None:
            woe_df = pd.DataFrame({f"{c}_woe": aplicar_woe(df[c], woes[c]) for c in CAT_PD}, index=df.index)
        return pd.concat([num, woe_df], axis=1)

    woe_tr = woe_train_fuera_de_muestra(X_train, y_train, seed) if WOE_FUERA_DE_MUESTRA else None
    return preparar(X_train, woe_tr), preparar(X_val), preparar(X_test)


NUM_EAD = ["funded_amnt", "interest_rate", "loan_term_months", "issue_date_year",
           "annual_income_log", "dept_paym_income_ratio",
           "num_30+_delinq_in_2yrs", "mths_since_last_delinq", "never_delinq", "num_derogatory_pub_rec",
           "num_open_credit_lines", "tot_num_credit_lines", "total_credit_revolving_bal",
           "used_credit_share", "credit_history_months",
           "num_inq_in_6mths", "num_inq_in_7_12mths",
           "num_open_trades_in_6mths", "num_installment_acc_op_in_24mths",
           "num_rev_trades_op_in_24mths", "max_bal_owed", "num_inq"]
CAT_EAD = ["home_ownership_status", "verification_status", "loan_purpose", "addr_state",
           "initial_list_status", "disbursement_method", "issue_date_month",
           "emp_length_cat", "emp_ambito",
           "mths_since_last_installment_acc_op_cat", "bal_to_cred_lim_cat",
           "grade"]
CAT_EXTRA_LGD = ["mths_since_last_delinq_cat", "num_derogatory_pub_rec_cat"]
COLS_PERDIDA = ["funded_amnt", "paym_rec", "princ_rec", "interest_rec", "late_fees_rec"]


def ead_observada(df):
    ead = (df["funded_amnt"] - df["princ_rec"]).clip(lower=0)
    return pd.DataFrame({"ead_obs": ead, "ead_pct": ead / df["funded_amnt"]}, index=df.index)


def lgd_observada(X, y):
    y = np.asarray(y).astype(int).ravel()
    d = X[COLS_PERDIDA].astype(float)
    o = pd.DataFrame(index=X.index)
    o["y"] = y
    o["ead"] = (d["funded_amnt"] - d["princ_rec"]).clip(lower=0)
    o["recuperacion"] = (d["paym_rec"] - d[["princ_rec", "interest_rec", "late_fees_rec"]].sum(axis=1)).clip(lower=0)
    o.loc[o["y"] == 0, "recuperacion"] = 0.0
    o["perdida"] = o["y"] * (o["ead"] - o["recuperacion"]).clip(lower=0)
    o["lgd"] = np.where((o["y"] == 1) & (o["ead"] > 0),
                        (o["perdida"] / o["ead"].where(o["ead"] > 0)).clip(0, 1), np.nan)
    return o


def matriz_ead(df, categorias):
    X = df[NUM_EAD].astype("float32").copy()
    for c in CAT_EAD:
        X[c] = pd.Categorical(df[c], categories=categorias[c])
    return X


# =====================================================================
# 2. MODELOS
# =====================================================================
SUAVIZADO_TE = 100


def con_nulos_str(s):
    return s.astype(object).where(s.notna(), "(nulo)").astype(str)


def ajustar_te(x, y, m=SUAVIZADO_TE):
    media_global = y.mean()
    stats = y.groupby(x).agg(["sum", "count"])
    return (stats["sum"] + m * media_global) / (stats["count"] + m), media_global


def aplicar_te(x, mapa, media_global):
    return x.map(mapa).fillna(media_global).astype(float)


class GLMFraccional:
    """Logística fraccional con target encoding fuera de muestra, imputación, indicadores de nulo,
    estandarización y año categórico. Todo se aprende en el X, y de fit()."""

    def __init__(self, num, cats, nulos=(), interaccion_tipo_año=False, n_bloques_te=5):
        self.num = list(num)
        self.cats = list(cats)
        self.nulos = list(nulos)
        self.interaccion = interaccion_tipo_año
        self.n_bloques_te = n_bloques_te

    def _numericas(self, X):
        d = X[[c for c in self.num if c != "funded_amnt_log"]].astype(float).copy()
        if "funded_amnt_log" in self.num:
            d["funded_amnt_log"] = np.log(X["funded_amnt"].astype(float))
        return d

    def _te_completo(self, X):
        return pd.DataFrame({f"te_{c}": aplicar_te(con_nulos_str(X[c]), *self.mapas_te[c]) for c in self.cats},
                            index=X.index)

    def _datos(self, X, te):
        num = self._numericas(X)
        d = pd.concat([num.fillna(self.medianas), te], axis=1)
        for nombre, col in self.nulos:
            d[nombre] = X[col].isna().astype(float)
        año = X["issue_date_year"].astype(int).astype(str)
        d["año"] = año.where(año.isin(self.niveles_año), self.niveles_año[0])
        d[self.cols_estandar] = (d[self.cols_estandar] - self.medias) / self.desv
        return d

    def fit(self, X, y, seed):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index, name="target")
        cats_tr = {c: con_nulos_str(X[c]) for c in self.cats}
        te_tr = pd.DataFrame(index=X.index, columns=[f"te_{c}" for c in self.cats], dtype=float)
        for i_aj, i_ap in KFold(self.n_bloques_te, shuffle=True, random_state=seed).split(X):
            i_aj, i_ap = X.index[i_aj], X.index[i_ap]
            for c in self.cats:
                mapa, mg = ajustar_te(cats_tr[c].loc[i_aj], y.loc[i_aj])
                te_tr.loc[i_ap, f"te_{c}"] = aplicar_te(cats_tr[c].loc[i_ap], mapa, mg).to_numpy()
        self.mapas_te = {c: ajustar_te(cats_tr[c], y) for c in self.cats}

        num_tr = self._numericas(X)
        self.medianas = num_tr.median()
        años = X["issue_date_year"].astype(int).astype(str)
        moda = años.mode()[0]
        self.niveles_año = [moda] + sorted(a for a in años.unique() if a != moda)
        self.cols_estandar = list(num_tr.columns) + list(te_tr.columns)
        base = pd.concat([num_tr.fillna(self.medianas), te_tr], axis=1)
        self.medias = base[self.cols_estandar].mean()
        self.desv = base[self.cols_estandar].std().replace(0, 1)

        datos = self._datos(X, te_tr)
        datos["target"] = y.to_numpy()
        termino_año = (f"interest_rate * C(año, levels={self.niveles_año!r})" if self.interaccion
                       else f"C(año, levels={self.niveles_año!r})")
        terminos = ([termino_año] + [f"Q('{c}')" for c in self.num if not (self.interaccion and c == "interest_rate")]
                    + [n for n, _ in self.nulos] + list(te_tr.columns))
        self.formula = "target ~ " + " + ".join(terminos)
        self.modelo = smf.glm(self.formula, data=datos, family=sm.families.Binomial()).fit(cov_type="HC0")
        assert self.modelo.converged, "La regresión fraccional no convergió"
        return self

    def predict(self, X):
        return pd.Series(self.modelo.predict(self._datos(X, self._te_completo(X))).to_numpy(), index=X.index)


NUM_EAD_M3 = ["loan_term_months", "num_open_credit_lines", "tot_num_credit_lines", "used_credit_share",
              "num_inq_in_6mths", "never_delinq", "max_bal_owed", "num_derogatory_pub_rec",
              "dept_paym_income_ratio", "credit_history_months", "num_rev_trades_op_in_24mths",
              "annual_income_log", "funded_amnt_log", "mths_since_last_delinq"]
CAT_EAD_M3 = ["bal_to_cred_lim_cat", "mths_since_last_installment_acc_op_cat", "issue_date_month",
              "emp_length_cat", "loan_purpose", "addr_state"]
NULOS_EAD_M3 = [("nulo_bloque_buro", "num_open_trades_in_6mths")]

NUM_LGD_M2 = ["loan_term_months", "interest_rate", "annual_income_log", "num_open_credit_lines",
              "tot_num_credit_lines", "funded_amnt_log", "num_rev_trades_op_in_24mths", "max_bal_owed"]
CAT_LGD_M2 = ["addr_state", "emp_ambito", "home_ownership_status", "verification_status",
              "emp_length_cat", "num_derogatory_pub_rec_cat"]

# ----------------------------- Modelo de PD -----------------------------
# Cada configuración guarda sus resultados en su propio CSV: resultados_semillas_<nombre>.csv.
# random_state NO va aquí: se fija con la semilla de cada vuelta dentro de pipeline().
_FIJOS_PD = dict(objective="binary", subsample_freq=1, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1)

MODELOS_PD = {
    # LightGBM linear trees, búsqueda con el WoE dentro de muestra (ensayo 38)
    "gbm_lt_ensayo38": dict(**_FIJOS_PD, linear_tree=True, n_estimators=4000,
                            learning_rate=0.010361, num_leaves=57, min_child_samples=363, linear_lambda=36.890709,
                            reg_lambda=0.001245, subsample=0.967926, colsample_bytree=0.539114),
    # LightGBM linear trees, búsqueda con el WoE fuera de muestra (ensayo 20)
    "gbm_lt_oof": dict(**_FIJOS_PD, linear_tree=True, n_estimators=4000,
                       learning_rate=0.011064856929825496, num_leaves=40, min_child_samples=433,
                       linear_lambda=0.722345183521419, reg_lambda=0.0011047270150509982,
                       subsample=0.9542974508089956, colsample_bytree=0.4490608665367354),
    # LightGBM estándar, búsqueda con el WoE fuera de muestra (ensayo 36) -> modelo final del notebook de desarrollo
    "gbm_estandar_oof": dict(**_FIJOS_PD, linear_tree=False, n_estimators=5000,
                             learning_rate=0.015442168320935673, num_leaves=40, min_child_samples=208,
                             reg_lambda=0.0010133406281001406, subsample=0.9443870282737901,
                             colsample_bytree=0.41804698005714136),
}

MODELO_PD = "gbm_lt_oof"          # <- configuración que se evalúa en esta ejecución
HIPERPARAMETROS_PD = MODELOS_PD[MODELO_PD]
RESULTADOS_CSV = os.path.join(RUTA_SCRIPT, f"resultados_semillas_{MODELO_PD}_conftotal{0.99}.csv")

ES_RONDAS_PD = 100


def predecir_cartera(X_pd, X_ead, X_lgd, modelo_pd, n_arboles_pd, modelo_ead, modelo_lgd):
    """PD, EAD y LGD predichas para una cartera y su ECL sin calibrar."""
    assert X_pd.index.equals(X_ead.index) and X_ead.index.equals(X_lgd.index), "Las matrices no están alineadas"
    c = pd.DataFrame(index=X_pd.index)
    c["pd"] = modelo_pd.predict_proba(X_pd, num_iteration=n_arboles_pd)[:, 1]
    c["funded"] = X_ead["funded_amnt"].astype(float)
    c["ead_pct_pred"] = modelo_ead.predict(X_ead).to_numpy()
    c["lgd_pred"] = modelo_lgd.predict(X_lgd).to_numpy()
    c["ead_usd_pred"] = c["ead_pct_pred"] * c["funded"]
    c["ecl_pred"] = c["pd"] * c["lgd_pred"] * c["ead_usd_pred"]
    return c


# =====================================================================
# 3. CALIBRACIÓN HÍBRIDA
# =====================================================================
COBERTURA = 0.90
EPS = 0.005
KAPPA_MIN = 1e-3
CONF_TOTAL = 0.99
B_DELTA = 100
THETA_Q = np.round(np.arange(0.70, 0.985, 0.005), 3)
PHIS = np.round(np.arange(0.80, 2.01, 0.02), 2)
Z_TOTAL = norm.ppf(CONF_TOTAL)


def entradas_calibracion(c):
    le = (c["lgd_pred"] * c["ead_usd_pred"]).to_numpy()
    return {"score": c["pd"].to_numpy() / le, "le": le,
            "techo": c["funded"].to_numpy(), "eb": c["ecl_pred"].to_numpy()}


def ecl_hibrida(score, le, techo, eb, theta, phi, kappa):
    alto = score >= theta
    return np.where(alto, np.minimum(phi * le, techo), kappa * eb)


def calibrar_hibrida(score, le, techo, eb, L, rng, n_test):
    m, L_tot = len(score), L.sum()
    factor = np.sqrt(1 + m / n_test)
    phi_req = np.where(techo > L, L / le, np.inf)

    cands = []
    for q in THETA_Q:
        theta = np.quantile(score, q)
        alto = score >= theta
        gratis_bajo = ((~alto) & (L == 0)).sum()
        cubiertos_alto = np.searchsorted(np.sort(phi_req[alto]), PHIS, side="left")
        cob = (gratis_bajo + cubiertos_alto) / m
        coste_alto = np.minimum(np.outer(le[alto], PHIS), techo[alto][:, None]).sum(axis=0)
        eb_bajo = eb[~alto].sum()
        kappa = np.maximum(KAPPA_MIN, (L_tot - coste_alto) / eb_bajo)
        for k in np.where(cob >= COBERTURA + EPS)[0]:
            cands.append((coste_alto[k] + kappa[k] * eb_bajo, theta, PHIS[k]))
    if not cands:
        raise ValueError("Ninguna combinación alcanza la cobertura: amplía THETA_Q o PHIS")

    total_min = min(cd[0] for cd in cands)
    mejor = None
    for total0, theta, phi in cands:
        if total0 > total_min * 1.001:
            continue
        alto = score >= theta
        e_alto = np.where(alto, np.minimum(phi * le, techo), 0.0)
        e_bajo = np.where(alto, 0.0, eb)
        k0 = max(KAPPA_MIN, (L_tot - e_alto.sum()) / e_bajo.sum())
        e0 = e_alto + k0 * e_bajo
        R = e0.sum() / L_tot
        sd = np.sqrt(m * np.var(e0 - R * L)) / L_tot
        delta_aprox = max(0.0, Z_TOTAL * factor * sd - (R - 1))
        k_final = max(KAPPA_MIN, ((1 + delta_aprox) * L_tot - e_alto.sum()) / e_bajo.sum())
        clave = (round(e_alto.sum() + k_final * e_bajo.sum(), -5), -k_final)
        if mejor is None or clave < mejor[0]:
            mejor = (clave, theta, phi)
    _, theta, phi = mejor

    alto = score >= theta
    e_alto = np.where(alto, np.minimum(phi * le, techo), 0.0)
    e_bajo = np.where(alto, 0.0, eb)
    k0 = max(KAPPA_MIN, (L_tot - e_alto.sum()) / e_bajo.sum())
    e0 = e_alto + k0 * e_bajo
    ratios = np.empty(B_DELTA)
    for b in range(B_DELTA):
        i = rng.integers(0, m, m)
        ratios[b] = e0[i].sum() / L[i].sum()
    R0 = e0.sum() / L_tot
    delta = max(0.0, Z_TOTAL * factor * ratios.std() - (R0 - 1))
    kappa = max(KAPPA_MIN, ((1 + delta) * L_tot - e_alto.sum()) / e_bajo.sum())
    return {"theta": float(theta), "phi": float(phi), "kappa": float(kappa), "delta": float(delta)}


# =====================================================================
# PIPELINE COMPLETO EN FUNCIÓN DE LA SEMILLA
# =====================================================================
def pipeline(seed, X, y):
    """Ejecuta todo el pipeline con la semilla dada y devuelve los resultados sobre test."""
    np.random.seed(seed)

    # --- Split y preprocesamiento ---
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, PERC_VAL, PERC_TEST, seed, STRATIFY)
    X_train, X_val, X_test = preprocesar(X_train, X_val, X_test)

    # --- Matrices ---
    X_train_pd, X_val_pd, X_test_pd = matrices_pd(X_train, X_val, X_test, y_train, seed)

    malos_train = X_train.loc[y_train == 1]
    categorias_ead = {c: sorted(malos_train[c].dropna().unique()) for c in CAT_EAD}
    X_train_EAD = matriz_ead(malos_train, categorias_ead)
    y_train_EAD = ead_observada(malos_train)["ead_pct"]
    X_val_EAD = matriz_ead(X_val, categorias_ead)
    X_test_EAD = matriz_ead(X_test, categorias_ead)

    lgd_obs_train = lgd_observada(X_train, y_train)
    lgd_obs_val = lgd_observada(X_val, y_val)
    idx_train_LGD = X_train_EAD.index[lgd_obs_train.loc[X_train_EAD.index, "ead"].to_numpy() > 0]
    X_train_LGD = X_train_EAD.loc[idx_train_LGD].copy()
    y_train_LGD = lgd_obs_train.loc[idx_train_LGD, "lgd"]
    X_val_LGD, X_test_LGD = X_val_EAD.copy(), X_test_EAD.copy()
    categorias_extra = {c: sorted(X_train.loc[idx_train_LGD, c].dropna().unique()) for c in CAT_EXTRA_LGD}
    for c in CAT_EXTRA_LGD:
        X_train_LGD[c] = pd.Categorical(X_train.loc[X_train_LGD.index, c], categories=categorias_extra[c])
        X_val_LGD[c] = pd.Categorical(X_val[c], categories=categorias_extra[c])
        X_test_LGD[c] = pd.Categorical(X_test[c], categories=categorias_extra[c])

    X_test_perdida = X_test[COLS_PERDIDA].copy()
    del X_train, X_val, X_test, malos_train

    # --- Entrenamiento ---
    modelo_ead = GLMFraccional(num=["interest_rate"] + NUM_EAD_M3, cats=CAT_EAD_M3, nulos=NULOS_EAD_M3,
                               interaccion_tipo_año=True).fit(X_train_EAD, y_train_EAD, seed)
    modelo_lgd = GLMFraccional(num=NUM_LGD_M2, cats=CAT_LGD_M2).fit(X_train_LGD, y_train_LGD, seed)

    i_fit, i_es = train_test_split(np.arange(len(y_train)), test_size=0.10, stratify=y_train, random_state=seed)
    y_tr = y_train.to_numpy()
    modelo_pd = lgb.LGBMClassifier(**HIPERPARAMETROS_PD, random_state=seed)
    modelo_pd.fit(X_train_pd.iloc[i_fit], y_tr[i_fit],
                  eval_set=[(X_train_pd.iloc[i_es], y_tr[i_es])], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(ES_RONDAS_PD, verbose=False)])
    n_arboles_pd = modelo_pd.best_iteration_

    # --- Calibración en validación ---
    cartera_val = predecir_cartera(X_val_pd, X_val_EAD, X_val_LGD, modelo_pd, n_arboles_pd, modelo_ead, modelo_lgd)
    perdida_val = lgd_obs_val.loc[cartera_val.index, "perdida"].to_numpy()
    rng = np.random.default_rng(seed)
    params = calibrar_hibrida(**entradas_calibracion(cartera_val), L=perdida_val, rng=rng, n_test=len(X_test_pd))

    # --- Test ---
    cartera_test = predecir_cartera(X_test_pd, X_test_EAD, X_test_LGD, modelo_pd, n_arboles_pd, modelo_ead, modelo_lgd)
    ecl_j = ecl_hibrida(**entradas_calibracion(cartera_test), **{k: params[k] for k in ("theta", "phi", "kappa")})
    real_j = lgd_observada(X_test_perdida, y_test).loc[cartera_test.index, "perdida"].to_numpy()

    ecl_total, real_total = ecl_j.sum(), real_j.sum()
    return pd.Series({
        "Modelo PD": MODELO_PD,
        "Semilla": seed,
        "Árboles PD": n_arboles_pd,
        "Restricción 1": bool(ecl_total > real_total),
        "Restricción 2": bool((ecl_j < real_j).mean() <= 0.10),
        "% ECL_j pred > ECL_j real": (ecl_j > real_j).mean(),
        "ECL total real": real_total,
        "ECL total predicha": ecl_total,
        "ECL predicha / ECL real": ecl_total / real_total,
    })


# =====================================================================
# BUCLE DE SEMILLAS
# =====================================================================
if __name__ == "__main__":
    print(f"Modelo de PD: {MODELO_PD} | resultados en {RESULTADOS_CSV}")
    print("Cargando datos...")
    X_TOTAL = pd.read_csv(os.path.join(RUTA_DATOS, "X.csv"), header=0)
    Y_TOTAL = pd.read_csv(os.path.join(RUTA_DATOS, "target.csv")).squeeze().rename("y").astype(int)
    assert len(X_TOTAL) == len(Y_TOTAL) and X_TOTAL.index.equals(Y_TOTAL.index)

    # Semillas ya evaluadas en una ejecución anterior: no se repiten
    resultados = []
    if os.path.exists(RESULTADOS_CSV):
        resultados = [fila for _, fila in pd.read_csv(RESULTADOS_CSV).iterrows()]
    hechas = {int(r["Semilla"]) for r in resultados}

    for s in SEMILLAS:
        if s in hechas:
            print(f"Semilla {s}: ya evaluada, se salta")
            continue
        t0 = time.time()
        print(f"\n===== Semilla {s} =====")
        r = pipeline(s, X_TOTAL, Y_TOTAL)
        r["Minutos"] = (time.time() - t0) / 60
        print(r.to_string())
        resultados.append(r)
        pd.DataFrame(resultados).to_csv(RESULTADOS_CSV, index=False)     # se guarda tras cada semilla

    tabla_semillas = pd.DataFrame(resultados).reset_index(drop=True)
    print("\n===== Resultados por semilla =====")
    print(tabla_semillas.to_string(index=False))
    print(f"\nCumplen ambas restricciones: "
          f"{(tabla_semillas['Restricción 1'].astype(bool) & tabla_semillas['Restricción 2'].astype(bool)).mean():.0%} "
          f"de las {len(tabla_semillas)} semillas")