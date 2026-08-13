"""
Práctica 1.1: Argumentos causales alternativos e incertidumbre.
============================================
"""

import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Definición del espacio de hipótesis
# ------------------------------------------------------------

# H representa las 3 cajas posibles {0, 1, 2}
H = np.arange(3)


# ------------------------------------------------------------
# 2. Priors
# ------------------------------------------------------------


def p_r(r: int) -> float:
    """
    P(R = r).
    Prior sobre la ubicación del regalo.
    r ∈ {0, 1, 2}
    """
    if r in H:
        return 1/3
    else:
        return 0


def p_c(c: int) -> float:
    """
    P(C = c).
    Prior sobre la caja elegida por el participante
    c ∈ {0, 1, 2}
    """
    if c in H:
        return 1/3
    else:
        return 0


# ------------------------------------------------------------
# 3. Modelos del presentador
# ------------------------------------------------------------


def p_s_rM0(s: int, r: int) -> float:
    """
    P(S = s | R= r, M = 0).
    Modelo 0 (No Monty Hall):
    El presentador abre cualquier caja que no tenga el regalo.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    """
    if s == r:
        return 0
    else:
        return 1/2


def p_s_rcM1(s: int, r: int, c: int):
    """
    P(S = s | R= r, C= c, M = 1).
    Modelo 1 (Monty Hall):
    El presentador abre una caja que no tenga el regalo
    ni haya sido seleccionada.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    """
    if s == r or s == c:
        return 0
    elif r == c:
        return 1/2
    else:
        return 1


# ------------------------------------------------------------
# 4. Distribución conjunta P(r, c, s | M)
# ------------------------------------------------------------


def p_rcs_M(r: int, c: int, s: int, m: int) -> float:
    """
    P(r, c, s | M) = P(r | M)P(c | M)P(s | r, c, M)
    Distribución conjunta del modelo m.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    m ∈ {0, 1}
    
    P(r) y P(c) son los mismos priors uniformes, cambia la prob de s
    """
    if m == 0:
        return p_r(r) * p_c(c) * p_s_rM0(s, r)
    elif m == 1:
        return p_r(r) * p_c(c) * p_s_rcM1(s, r, c)


# ------------------------------------------------------------
# 5. Simulación de datos (asumiendo Monty Hall verdadero)
# ------------------------------------------------------------

np.random.seed(0)


def simular(T=16) -> List[Tuple[int, int, int]]:
    """
    Función para simular datos según el modelo Monty Hall verdadero.
    T: número de datos a generar.

    forma compacta de escribir la lista p=[p_r(r) for r in H], es como hacer un for y append
    ncestral sampling: primero generamos r y c, y recién después, usando esos valores, generamos s.
    
    en cada paso del for se hace el append en la lista datos
    """
    datos = []

    for t in range(T):
        r = np.random.choice(H, p=[p_r(r) for r in H])
        c = np.random.choice(H, p=[p_c(c) for c in H])
        s = np.random.choice(H, p=[p_s_rcM1(s, r, c) for s in H])
        datos.append((c, s, r))

    return datos
# ------------------------------------------------------------
# 6. Probabilidad de los datos dado un modelo
# ------------------------------------------------------------

def pDatos_M(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    P(Datos | M) = prod([ P(c,s,r|M) for c, s, r in Datos ])
    Probabilidad de ver los datos dados el modelo considerado.
    m ∈ {0, 1}

    P(Datos|M)=P(d1|M)*P(d2|M)*...

    ojo en el for giro las variables
    """
    probabilidad = 1

    for c, s, r in datos:
        probabilidad = probabilidad * p_rcs_M(r, c, s, m)

    return probabilidad


# ------------------------------------------------------------
# 7. Probabilidad total de ver los datos
# ------------------------------------------------------------


def pM(m: int) -> float:
    """
    P(M)
    Prior sobre los modelos.
    m ∈ {0, 1}

    Cuánto creemos inicialmente en cada modelo: por máxima incertidumbre,
    no favorecemos ningún modelo.
    creencia del modelo antes de ver los datos

    """
    return 1/2


def pDatos(datos: List[Tuple[int, int, int]]) -> float:
    """
    P(Datos)
    Probabilidad total de ver los datos usando la contribución
    de todos los modelos.

    P(Datos)=P(Datos|M0)P(M0)+P(Datos|M1)P(M1)
    """
    return pDatos_M(datos, 0) * pM(0) + pDatos_M(datos, 1) * pM(1)


# ------------------------------------------------------------
# 8. Actualización de creencias sobre los modelos
# ------------------------------------------------------------


def pM_Datos(m: int, datos: List[Tuple[int, int, int]]) -> float:
    """
    P(M | Datos)
    Actualización de la creencia sobre el modelo M dado los datos
    (distribución a posteriori).

    Bayes, creencia del modelo después de ver los datos
    """
    return pDatos_M(datos, m) * pM(m) / pDatos(datos)


def evolucion_posterior(m: int, datos: List[Tuple[int, int, int]]) -> List[float]:
    """
    Evolución del posterior de los modelos según se observa información.
    
    Vamos sumando episodios de a uno y analizamos como evoluciona el posterior
    """
    posteriores = []

    for t in range(len(datos) + 1):
        posteriores.append(pM_Datos(m, datos[:t]))

    return posteriores



# ------------------------------------------------------------
# 9. Cálculo del bayes factor
# (diferencia en órdenes de magnitud en escala logarítmica)
# ------------------------------------------------------------


def log_pDatos_M(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    log(P(Datos | M))
    Logaritmo de la probabilidad de ver los datos dados el modelo considerado.
    """
    log_probabilidad = 0

    for c, s, r in datos:
        log_probabilidad = log_probabilidad + np.log10(p_rcs_M(r, c, s, m))

    return log_probabilidad


def log_bayes_factor(datos: List[Tuple[int, int, int]], m_i: int, m_j: int) -> float:
    """
    log(Datos | Mi / Datos | Mj) = log_10P(Datos | Mi) - log_10P(Datos | Mj)
    Bayes factor en escala logarítmica para comparar dos modelos.
    m_i, m_j ∈ {0, 1}
    """
    return log_pDatos_M(datos, m_i) - log_pDatos_M(datos, m_j)


# ------------------------------------------------------------
# 10. Predicción típica que realizan los modelos sobre los datos
# ------------------------------------------------------------

def prediccion_tipica(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    log(Media Geométrica)
    Logaritmo de la media geométrica de los datos dados el modelo considerado.
    m ∈ {0, 1}
    """
    N = len(datos)
    return 10 ** (log_pDatos_M(datos, m) / N)



# ------------------------------------------------------------
# 11. Ejecución principal
# ------------------------------------------------------------
if __name__ == "__main__":
    # Simulación
    datos = simular(T=16)

    # Print de funciones
    print("log_bayes_factor (M1 contra M0):", log_bayes_factor(datos, 1, 0))
    print("prediccion_tipica (M0):", prediccion_tipica(datos, 0))
    print("prediccion_tipica (M1):", prediccion_tipica(datos, 1))

    # Posteriores
    post_M0 = evolucion_posterior(0, datos)
    post_M1 = evolucion_posterior(1, datos)

    # Gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(post_M0, label="M0: Base")
    plt.plot(post_M1, label="M1: Monty Hall")
    plt.xlabel("Número de episodios")
    plt.ylabel("P(Modelo | Datos)")
    plt.title("Evolución del posterior de los modelos")
    plt.legend()
    plt.tight_layout()

    # Guardar el gráfico en lugar de abrir una ventana
    plt.savefig("posterior.png")
    plt.close()

    print("Gráfico guardado como posterior.png")