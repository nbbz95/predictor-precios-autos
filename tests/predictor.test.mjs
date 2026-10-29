// Verifica que la implementación en JS dé lo mismo que scikit-learn para los casos exportados.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { predecir } from '../web/predictor.js'

const modelo = JSON.parse(readFileSync(new URL('../web/modelo.json', import.meta.url)))

test('coincide con scikit-learn en los casos de prueba', () => {
  assert.ok(modelo.casos_de_prueba.length >= 5)
  for (const { entrada, esperado } of modelo.casos_de_prueba) {
    const obtenido = predecir(modelo, entrada)
    assert.ok(Math.abs(obtenido - esperado) / esperado < 1e-6, `${obtenido} vs ${esperado}`)
  }
})

test('usa la mediana si falta un valor numérico', () => {
  const base = { ...modelo.casos_de_prueba[0].entrada }
  delete base.horsepower
  assert.ok(Number.isFinite(predecir(modelo, base)))
})

test('una marca desconocida no rompe la predicción', () => {
  const entrada = { ...modelo.casos_de_prueba[0].entrada, make: 'tesla' }
  assert.ok(predecir(modelo, entrada) > 0)
})
