from modelo.entrenar import CATEGORICAS, NUMERICAS, OBJETIVO, candidatos, cargar, exportar_ridge


def test_cargar_descarta_autos_sin_precio():
    df = cargar()
    assert df[OBJETIVO].notna().all()
    assert set(NUMERICAS + CATEGORICAS) <= set(df.columns)


def test_exportacion_reproduce_las_predicciones():
    df = cargar()
    modelo = candidatos()["Ridge sobre log(precio)"].fit(df.drop(columns=OBJETIVO), df[OBJETIVO])
    exp = exportar_ridge(modelo, df)
    assert set(exp["numericas"]) == set(NUMERICAS)
    assert set(exp["categoricas"]) == set(CATEGORICAS)
    assert all(c["esperado"] > 0 for c in exp["casos_de_prueba"])
