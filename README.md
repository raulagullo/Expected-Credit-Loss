# Expected Credit Loss de una cartera de préstamos (Lending Club)

Estimación de la pérdida esperada (ECL) de una cartera de más de 2 millones de préstamos de Lending Club, préstamo a préstamo, con sesgo conservador:

$$\widehat{ECL}_j = \widehat{PD}_j \cdot \widehat{LGD}_j \cdot \widehat{EAD}_j$$

Proyecto desarrollado para el reto mensual de Data Science de la comunidad *Outliers In Data* (octubre de 2026).

## El reto

A partir de las variables disponibles en el momento de conceder cada préstamo (variables de originación), estimar la ECL de cada préstamo de una cartera de test, de modo que:

1. **La ECL total estimada supere a la pérdida real total** de la cartera.
2. **La ECL estimada sea inferior a la pérdida real en, como mucho, el 10 % de los préstamos.**

Cumpliendo ambas restricciones, se busca la menor ECL total posible. Las bases prohíben usar variables de desempeño o `grade` en el modelo de PD, excluir filas a mano y utilizar el conjunto de test para cualquier decisión de modelización.

## Enfoque

| Componente | Modelo seleccionado | Test |
|---|---|---|
| **PD** (probabilidad de impago) | LightGBM optimizado con Optuna (validación cruzada dentro de train), con WoE calculado fuera de muestra para las categóricas | AUC 0.754 · log-loss 0.339 · PD media 13.02 % frente a una tasa real del 13.08 % |
| **EAD** (exposición en el impago) | Regresión logística fraccional sobre la fracción del principal pendiente | EAD total predicha / real 0.9994 |
| **LGD** (pérdida en caso de impago) | Regresión logística fraccional | Pérdida total predicha / real 1.0030 |

Se compararon además regresiones logísticas, linear trees y LightGBM con árboles lineales; los modelos más complejos aportaban mejoras marginales que no compensaban la pérdida de simplicidad.

**Calibración híbrida.** Con $\widehat{PD}_j < 1$, la ECL de casi todos los impagos queda por debajo de su pérdida, y un factor de corrección único tendría que multiplicar la ECL por casi 3 para cumplir la cobertura. La calibración divide la cartera en dos tramos según los impagos esperados por dólar de provisión:

- **Tramo alto:** se provisiona como si el préstamo fuera a impagar, $\min(\varphi \cdot \widehat{LGD}_j \widehat{EAD}_j,\ \text{importe concedido})$.
- **Tramo bajo:** la ECL de los modelos se escala por un factor $\kappa$, ajustado para que la ECL total supere a la pérdida con un margen $\delta$ dimensionado estadísticamente (método delta y bootstrap, confianza del 99,9 %).

## Resultados en test

Todas las decisiones de modelización y los parámetros de la calibración se fijaron con train y validación. El conjunto de test (427.929 préstamos) se usó una única vez, al final, aplicando el pipeline congelado.

| | ECL predicha | Pérdida real | ECL / pérdida | Préstamos infraestimados | MAE ECL$_j$ |
|---|---:|---:|---:|---:|---:|
| **Test, calibrada** | 566.958.326 $ | 556.332.939 $ | **1.0191** | **9.50 %** | 2.128 $ |
| Test, sin calibrar | 556.159.355 $ | 556.332.939 $ | 0.9997 | 12.81 % | 2.001 $ |
| Validación, calibrada | 566.779.571 $ | 554.073.730 $ | 1.0229 | 9.47 % | 2.127 $ |

**Se cumplen las dos restricciones del reto**: la ECL total supera a la pérdida real en un 1.91 % y solo el 9.50 % de los préstamos queda infraestimado. Sin calibrar, el producto de los tres modelos incumple ambas. Los resultados en test son prácticamente iguales a los de validación, lo que indica que la calibración no está sobreajustada a la muestra en la que se aprendió; además, una comprobación *out-of-bag* en validación cumplía ambas restricciones en el 99 % de las réplicas.

El MAPE de ECL$_j$ en test, calculado sobre los préstamos con pérdida positiva, es del 155.20 % (104.98 % excluyendo un único impago con exposición residual inferior a 1 $). La calibración prioriza cumplir las restricciones con la menor ECL total, no el ajuste préstamo a préstamo.

## Estructura del repositorio

```
├── Expected_Credit_Loss.ipynb     # notebook completo: EDA, modelización, calibración y conclusiones
├── emp_title_clasification.py     # clasificador de puestos de trabajo (emp_title) usado en el preprocesamiento
├── modelos/                       # modelos optimizados entrenados en local y cargados desde el notebook
├── scripts/                       # búsquedas de Optuna y prueba de robustez con varias semillas
├── requirements.txt
└── data/                          # X.csv y target.csv (no incluidos)
```

## Reproducibilidad

- El notebook se ejecutó en **Google Colab**. Las búsquedas de hiperparámetros con Optuna y el entrenamiento final de los modelos optimizados se realizaron en local con los scripts de `scripts/`; su código se muestra en el notebook como referencia y los modelos resultantes se cargan desde `modelos/`.
- Para ejecutarlo en otro entorno hay que sustituir las rutas de Google Drive por rutas locales (`data/` y `modelos/`) y eliminar el montaje de Drive.
- Toda la aleatoriedad depende de una única variable `SEED`.
- Los datos del reto no se incluyen en el repositorio.

```
pip install -r requirements.txt
```

## Limitaciones principales

- El target es el impago **observado hasta la fecha de corte** de los datos: las cosechas recientes aún no han madurado.
- Train, validación y test proceden de una partición aleatoria del mismo periodo; no se ha validado sobre préstamos de periodos posteriores (*out-of-time*).
- La LGD usa recuperaciones en valor nominal, sin descuento ni costes de recobro, por falta de datos.

El detalle de supuestos, limitaciones y líneas futuras está en la sección de conclusiones del notebook.

## Autor

Raúl Agulló · [LinkedIn](https://www.linkedin.com/in/ra%C3%BAl-agull%C3%B3-garc%C3%ADa)
