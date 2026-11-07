import os
import random

FILAS = 4
COLUMNAS = 12
ACCIONES = ("N", "E", "S", "W")
MOVIMIENTOS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
FLECHAS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}

INICIO = (3, 0)
META = (3, 11)
ACANTILADO = {(3, c) for c in range(1, 11)}


def paso_entorno(estado, accion):
    f, c = estado
    df, dc = MOVIMIENTOS[accion]
    nf, nc = f + df, c + dc

    if nf < 0 or nf >= FILAS or nc < 0 or nc >= COLUMNAS:
        nf, nc = f, c

    sig_estado = (nf, nc)

    if sig_estado in ACANTILADO:
        return INICIO, -100, False

    if sig_estado == META:
        return META, -1, True

    return sig_estado, -1, False


def crear_tabla_q():
    Q = {}
    for f in range(FILAS):
        for c in range(COLUMNAS):
            if (f, c) != META:
                Q[(f, c)] = {a: 0.0 for a in ACCIONES}
    return Q


def mejor_accion(Q, estado):
    mejor_a = ACCIONES[0]
    mejor_val = Q[estado][mejor_a]
    for a in ACCIONES[1:]:
        if Q[estado][a] > mejor_val:
            mejor_a = a
            mejor_val = Q[estado][a]
    return mejor_a


def elegir_accion_epsilon_greedy(Q, estado, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(ACCIONES)
    return mejor_accion(Q, estado)


def entrenar_q_learning(episodios=500, alpha=0.5, gamma=1.0, epsilon=0.1, max_pasos=200, semilla=42):
    rng = random.Random(semilla)
    Q = crear_tabla_q()
    recompensas_por_episodio = []

    for _ in range(episodios):
        estado = INICIO
        retorno_ep = 0

        for _ in range(max_pasos):
            accion = elegir_accion_epsilon_greedy(Q, estado, epsilon, rng)
            sig_estado, recompensa, terminado = paso_entorno(estado, accion)

            if terminado:
                max_q_sig = 0.0
            else:
                max_q_sig = max(Q[sig_estado].values())

            q_actual = Q[estado][accion]
            Q[estado][accion] = q_actual + alpha * (recompensa + gamma * max_q_sig - q_actual)

            retorno_ep += recompensa
            estado = sig_estado

            if terminado:
                break

        recompensas_por_episodio.append(retorno_ep)

    return Q, recompensas_por_episodio


def evaluar_politica_greedy(Q, max_pasos=200):
    estado = INICIO
    camino = [estado]
    acciones_tomadas = []
    retorno_total = 0

    for _ in range(max_pasos):
        accion = mejor_accion(Q, estado)
        sig_estado, recompensa, terminado = paso_entorno(estado, accion)

        acciones_tomadas.append(accion)
        camino.append(sig_estado)
        retorno_total += recompensa
        estado = sig_estado

        if terminado:
            break

    return camino, acciones_tomadas, retorno_total


def imprimir_cuadricula_politica(Q):
    for f in range(FILAS):
        fila_str = []
        for c in range(COLUMNAS):
            pos = (f, c)
            if pos == META:
                celda = " G "
            elif pos == INICIO:
                celda = "S" + FLECHAS[mejor_accion(Q, pos)] + " "
            elif pos in ACANTILADO:
                celda = " C "
            else:
                celda = " " + FLECHAS[mejor_accion(Q, pos)] + " "
            fila_str.append(celda)
        print(" ".join(fila_str))


def calcular_media_movil(valores, ventana=10):
    medias = []
    for i in range(len(valores)):
        inicio_v = max(0, i - ventana + 1)
        tramo = valores[inicio_v : i + 1]
        medias.append(sum(tramo) / len(tramo))
    return medias


def guardar_grafica(recompensas, ventana=10, archivo="curva_recompensas.png"):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Nota: matplotlib no esta instalado en este entorno; se conserva curva_recompensas.png existente.")
        return

    medias_moviles = calcular_media_movil(recompensas, ventana)
    episodios_x = list(range(1, len(recompensas) + 1))

    plt.figure(figsize=(9, 4.5))
    plt.plot(episodios_x, recompensas, color="lightgray", linewidth=1, label="Recompensa por episodio")
    plt.plot(episodios_x, medias_moviles, color="tab:red", linewidth=1.8, label=f"Media movil ({ventana} episodios)")
    plt.axhline(-13, color="tab:green", linestyle="--", linewidth=1.2, label="Retorno greedy optimo (-13)")

    plt.title("Recompensa por episodio - Q-learning en Cliff Walking")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")
    plt.ylim(-150, 0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    ruta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), archivo)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"Grafica guardada en: {archivo}")


if __name__ == "__main__":
    assert paso_entorno((3, 0), "N") == ((2, 0), -1, False)
    assert paso_entorno((3, 0), "E") == ((3, 0), -100, False)
    assert paso_entorno((2, 11), "S") == ((3, 11), -1, True)

    SEMILLA = 42
    ALPHA = 0.5
    GAMMA = 1.0
    EPSILON = 0.1
    EPISODIOS = 500
    MAX_PASOS = 200

    Q, recompensas = entrenar_q_learning(
        episodios=EPISODIOS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        max_pasos=MAX_PASOS,
        semilla=SEMILLA,
    )

    print("=== Q-learning en Cliff Walking (4x12) ===")
    print(f"Parametros: alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, episodios={EPISODIOS}, semilla={SEMILLA}")
    print(f"Orden de desempate en argmax: {ACCIONES}\n")

    print("Politica greedy aprendida sobre la cuadricula:")
    imprimir_cuadricula_politica(Q)
    print()

    camino, acciones, retorno_greedy = evaluar_politica_greedy(Q)
    print("Evaluacion greedy (epsilon = 0):")
    print(f"  Acciones ({len(acciones)} pasos): {' -> '.join(acciones)}")
    print(f"  Camino: {' -> '.join(str(s) for s in camino)}")
    print(f"  Numero de pasos: {len(acciones)}")
    print(f"  Retorno greedy: {retorno_greedy}\n")

    medias = calcular_media_movil(recompensas, ventana=10)
    print("Recompensa por episodio durante el entrenamiento (muestra cada 50 episodios):")
    print("  Episodio | Recompensa ep. | Media movil (10 ep.)")
    for ep in [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        idx = ep - 1
        print(f"  {ep:8d} | {recompensas[idx]:14d} | {medias[idx]:20.2f}")
    print()

    guardar_grafica(recompensas)
