# Predictor de precios de autos

Modelo de machine learning que estima el precio de un auto a partir de sus características. Se entrena con scikit-learn y se exporta a JSON para correr en el navegador sin servidor.

**Demo:** https://nbbz95.github.io/predictor-precios-autos/

## Resultados

Validación cruzada de 5 folds sobre 201 autos:

| Modelo | R² | Error medio | Error % |
|---|---|---|---|
| Regresión lineal | 0,856 | USD 1.845 | 14,3% |
| **Ridge sobre log(precio)** | **0,910** | **USD 1.422** | **10,4%** |
| Random Forest | 0,884 | USD 1.585 | 11,1% |
| Gradient Boosting | 0,887 | USD 1.513 | 11,0% |

Con un dataset chico (201 filas), un modelo lineal regularizado sobre el logaritmo del precio le gana a los ensambles de árboles: el log corrige la cola larga de precios y Ridge evita el sobreajuste con las marcas que tienen pocos autos.

## Pipeline

1. **Datos:** [Automobile de UCI](https://archive.ics.uci.edu/dataset/10/automobile) (CC BY 4.0), en `data/`. Se descartan los 4 autos sin precio.
2. **Variables:** 7 numéricas (cilindrada, potencia, peso, medidas, consumo) y 5 categóricas (marca, combustible, aspiración, carrocería, tracción).
3. **Preprocesamiento:** imputación por mediana, estandarización y one-hot encoding dentro de un `Pipeline`, para que no haya fuga de datos entre folds.
4. **Comparación:** 4 modelos con `cross_validate` ([`modelo/entrenar.py`](modelo/entrenar.py)).
5. **Exportación:** los coeficientes del Ridge, las medias y escalas se guardan en `web/modelo.json`. [`web/predictor.js`](web/predictor.js) los evalúa en unas 15 líneas.
6. **Verificación:** un test en Node compara las predicciones de JS con las de scikit-learn (tolerancia 1e-6).

El CI reentrena el modelo, corre los tests de Python y de JS y publica en GitHub Pages en cada push.

## Correr localmente

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python -m modelo.entrenar      # compara modelos y exporta web/modelo.json
npm test                       # JS vs scikit-learn
python -m http.server -d web 8000
```

## Limitaciones

Los precios son de autos del mercado de EE.UU. de 1985, en dólares de esa época. El objetivo del proyecto es mostrar el flujo de ML de punta a punta, no tasar autos actuales.

## Stack

Python, pandas, scikit-learn, JavaScript, Node test runner, GitHub Actions y GitHub Pages.
