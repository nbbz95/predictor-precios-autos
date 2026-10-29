"""Entrena y compara modelos de precio, y exporta el elegido a JSON para usarlo en el navegador."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "imports-85.csv"

NUMERICAS = ["engine-size", "horsepower", "curb-weight", "width", "length", "city-mpg", "highway-mpg"]
CATEGORICAS = ["make", "fuel-type", "aspiration", "body-style", "drive-wheels"]
OBJETIVO = "price"


def cargar(ruta=DATOS):
    df = pd.read_csv(ruta, na_values=["NA", "?"])
    return df.dropna(subset=[OBJETIVO])[NUMERICAS + CATEGORICAS + [OBJETIVO]]


def preprocesador(escalar=True):
    numericas = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        numericas.append(("escalar", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numericas), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )


def candidatos():
    """El precio tiene cola larga: los modelos lineales se entrenan sobre log(precio)."""

    def en_log(modelo):
        return TransformedTargetRegressor(regressor=modelo, func=np.log, inverse_func=np.exp)

    return {
        "Regresión lineal": Pipeline([("pre", preprocesador()), ("m", LinearRegression())]),
        "Ridge sobre log(precio)": en_log(
            Pipeline([("pre", preprocesador()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))])
        ),
        "Random Forest": Pipeline(
            [("pre", preprocesador(False)), ("m", RandomForestRegressor(n_estimators=300, random_state=42))]
        ),
        "Gradient Boosting": Pipeline(
            [("pre", preprocesador(False)), ("m", GradientBoostingRegressor(random_state=42))]
        ),
    }


def comparar(df):
    X, y = df.drop(columns=OBJETIVO), df[OBJETIVO]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    filas = []
    for nombre, modelo in candidatos().items():
        r = cross_validate(modelo, X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error", "neg_mean_absolute_percentage_error"))
        filas.append(
            {
                "modelo": nombre,
                "r2": round(r["test_r2"].mean(), 3),
                "mae": round(-r["test_neg_mean_absolute_error"].mean()),
                "mape_pct": round(-r["test_neg_mean_absolute_percentage_error"].mean() * 100, 1),
            }
        )
    return filas


def exportar_ridge(modelo, df, n_casos=10):
    """Pasa el Ridge entrenado a coeficientes planos que el navegador puede evaluar sin librerías."""
    pipe = modelo.regressor_
    pre, ridge = pipe.named_steps["pre"], pipe.named_steps["m"]
    num = pre.named_transformers_["num"]
    imputer, scaler = num.named_steps["imputar"], num.named_steps["escalar"]
    onehot = pre.named_transformers_["cat"]
    coef = ridge.coef_
    k = len(NUMERICAS)

    exportado = {
        "intercepto": float(ridge.intercept_),
        "numericas": {
            c: {
                "mediana": float(imputer.statistics_[i]),
                "media": float(scaler.mean_[i]),
                "escala": float(scaler.scale_[i]),
                "coef": float(coef[i]),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
            }
            for i, c in enumerate(NUMERICAS)
        },
        "categoricas": {},
        "alpha": float(ridge.alpha_),
    }
    pos = k
    for c, categorias in zip(CATEGORICAS, onehot.categories_):
        exportado["categoricas"][c] = {str(cat): float(coef[pos + j]) for j, cat in enumerate(categorias)}
        pos += len(categorias)

    muestra = df.drop(columns=OBJETIVO).sample(n_casos, random_state=0)
    exportado["casos_de_prueba"] = [
        {"entrada": fila.dropna().to_dict(), "esperado": float(p)}
        for (_, fila), p in zip(muestra.iterrows(), modelo.predict(muestra))
    ]
    return exportado


def importancia(exportado):
    """Efecto de mover cada variable numérica un desvío estándar, en % sobre el precio."""
    return sorted(
        ({"variable": c, "efecto_pct": round((np.exp(v["coef"]) - 1) * 100, 1)} for c, v in exportado["numericas"].items()),
        key=lambda x: -abs(x["efecto_pct"]),
    )


def main():
    df = cargar()
    metricas = comparar(df)
    for m in metricas:
        print(f"{m['modelo']:<26} R²={m['r2']:.3f}  MAE=${m['mae']:,}  MAPE={m['mape_pct']}%")

    modelo = candidatos()["Ridge sobre log(precio)"].fit(df.drop(columns=OBJETIVO), df[OBJETIVO])
    exportado = exportar_ridge(modelo, df)
    exportado["metricas"] = metricas
    exportado["importancia"] = importancia(exportado)
    exportado["n_autos"] = len(df)
    salida = RAIZ / "web" / "modelo.json"
    salida.write_text(json.dumps(exportado, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Modelo exportado a {salida}")


if __name__ == "__main__":
    main()
