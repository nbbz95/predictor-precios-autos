// Evalúa el modelo Ridge exportado desde Python. Sin dependencias: corre igual en el navegador y en Node.

export function predecir(modelo, entrada) {
  let z = modelo.intercepto
  for (const [nombre, p] of Object.entries(modelo.numericas)) {
    const valor = Number.isFinite(Number(entrada[nombre])) && entrada[nombre] !== '' ? Number(entrada[nombre]) : p.mediana
    z += ((valor - p.media) / p.escala) * p.coef
  }
  for (const [nombre, categorias] of Object.entries(modelo.categoricas)) {
    // Una categoría desconocida aporta 0, igual que OneHotEncoder(handle_unknown="ignore").
    z += categorias[entrada[nombre]] ?? 0
  }
  return Math.exp(z)
}
