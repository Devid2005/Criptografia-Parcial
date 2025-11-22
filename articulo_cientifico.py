import os
import struct
import random
from math import sqrt, log2
import csv
import matplotlib.pyplot as plt

# =====================================================
# 1. Utilidades de ChaCha
# =====================================================
# Esta sección implementa las operaciones básicas del cifrador
# ChaCha, incluyendo rotación, quarter-round y las double-rounds
# completas. Se sigue la especificación estándar descrita por
# Bernstein en el diseño de ChaCha20.


def rotl32(x, n):
    """
    Realiza una rotación circular a la izquierda sobre 32 bits.

    Parámetros
    ----------
    x : int
        Entero de 32 bits sobre el que se rota.
    n : int
        Número de bits a rotar.

    Retorna
    -------
    int :
        Resultado de aplicar la rotación circular.
    """
    return ((x << n) & 0xffffffff) | (x >> (32 - n))


def quarter_round(a, b, c, d):
    """
    Implementa un quarter-round de ChaCha.

    Este bloque opera sobre cuatro palabras de 32 bits y es el
    componente básico del mezclado de ChaCha. Garantiza difusión
    y no linealidad mediante sumas módulo 2^32, XOR y rotaciones.

    Parámetros
    ----------
    a, b, c, d : ints
        Palabras de 32 bits del estado interno.

    Retorna
    -------
    tuple :
        Cuatro palabras de 32 bits actualizadas.
    """
    # Operaciones estándar del diseño ChaCha
    a = (a + b) & 0xffffffff
    d ^= a
    d = rotl32(d, 16)

    c = (c + d) & 0xffffffff
    b ^= c
    b = rotl32(b, 12)

    a = (a + b) & 0xffffffff
    d ^= a
    d = rotl32(d, 8)

    c = (c + d) & 0xffffffff
    b ^= c
    b = rotl32(b, 7)

    return a, b, c, d


def chacha_rounds(state, rounds=3):
    """
    Aplica 'rounds' double-rounds de ChaCha a un estado de 16 palabras.

    Cada double-round consta de:
    - una column round
    - una diagonal round

    Esta función permite reducir el número de rondas para estudiar
    versiones debilitadas del cifrador, como en distingidores y
    análisis estadísticos.

    Parámetros
    ----------
    state : list[int]
        Lista de 16 palabras de 32 bits.
    rounds : int
        Número de double-rounds (ChaCha20 usa rounds=10).

    Retorna
    -------
    list[int] :
        El estado actualizado después de aplicar las rondas.
    """
    x = list(state)
    for _ in range(rounds):
        # Column round
        x[0], x[4],  x[8],  x[12] = quarter_round(x[0], x[4],  x[8],  x[12])
        x[1], x[5],  x[9],  x[13] = quarter_round(x[1], x[5],  x[9],  x[13])
        x[2], x[6],  x[10], x[14] = quarter_round(x[2], x[6],  x[10], x[14])
        x[3], x[7],  x[11], x[15] = quarter_round(x[3], x[7],  x[11], x[15])

        # Diagonal round
        x[0], x[5],  x[10], x[15] = quarter_round(x[0], x[5],  x[10], x[15])
        x[1], x[6],  x[11], x[12] = quarter_round(x[1], x[6],  x[11], x[12])
        x[2], x[7],  x[8],  x[13] = quarter_round(x[2], x[7],  x[8],  x[13])
        x[3], x[4],  x[9],  x[14] = quarter_round(x[3], x[4],  x[9],  x[14])

    return x


def chacha_init_state(key, counter, nonce):
    """
    Construye el estado inicial de ChaCha (16 palabras de 32 bits).

    Este estado sigue la estructura:
    [constants | key | key | counter | nonce]

    Parámetros
    ----------
    key : bytes
        Clave de 256 bits (32 bytes).
    counter : int
        Contador de 32 bits.
    nonce : bytes
        Número aleatorio de 96 bits (12 bytes).

    Retorna
    -------
    list[int] :
        Lista de 16 palabras de 32 bits que forman el estado inicial.
    """
    constants = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]

    k_words = list(struct.unpack("<8I", key))     # 8 palabras de 32 bits
    n_words = list(struct.unpack("<3I", nonce))   # 3 palabras de 32 bits

    return [
        constants[0], constants[1], constants[2], constants[3],
        k_words[0], k_words[1], k_words[2], k_words[3],
        k_words[4], k_words[5], k_words[6], k_words[7],
        counter, n_words[0], n_words[1], n_words[2]
    ]


# =====================================================
# 2. Experimento estadístico simplificado
# =====================================================

def experiment_chacha_dl(
    key,
    rounds=3,
    n_pairs=1_000_000,
    diff_word_index=14,
    diff_value=0x00000040,
    output_word_indices=(3, 4),
    bit_position=0,
    verbose=True
):
    """
    Realiza un experimento estadístico sobre ChaCha reducido.

    Este experimento genera pares de estados idénticos salvo por
    una diferencia controlada en una palabra del estado inicial.
    Tras ejecutar ChaCha unas pocas rondas, se evalúa un evento
    estadístico muy simple basado en la paridad de ciertos bits de
    dos palabras de salida.

    El objetivo es buscar desviaciones respecto a p=1/2 que puedan
    sugerir la existencia de un distinguidor.

    Parámetros
    ----------
    key : bytes
        Clave fija usada en todos los pares.
    rounds : int
        Número de double-rounds a aplicar.
    n_pairs : int
        Número de pares a generar.
    diff_word_index : int
        Índice de la palabra donde se introduce la diferencia.
    diff_value : int
        Valor de la diferencia XOR aplicada.
    output_word_indices : tuple[int]
        Índices de las palabras de salida donde se evaluará el bit.
    bit_position : int
        Bit dentro de cada palabra que se evaluará.
    verbose : bool
        Si True, imprime los resultados del experimento.

    Retorna
    -------
    tuple :
        (p_hat, bias, sigma)
        donde:
        - p_hat = prob. empírica del evento
        - bias  = |p_hat - 1/2|
        - sigma = desviación estándar estadística esperada
    """

    count_event = 0

    for _ in range(n_pairs):
        counter = random.getrandbits(32)
        nonce = os.urandom(12)

        # Estado original
        s0 = chacha_init_state(key, counter, nonce)

        # Estado con diferencia en diff_word_index
        s1 = list(s0)
        s1[diff_word_index] ^= diff_value

        # Ejecutar ChaCha reducido
        y0 = chacha_rounds(s0, rounds=rounds)
        y1 = chacha_rounds(s1, rounds=rounds)

        # Verificar paridad XOR de los bits seleccionados
        def parity_out(state):
            p = 0
            for idx in output_word_indices:
                p ^= ((state[idx] >> bit_position) & 1)
            return p

        if parity_out(y0) == parity_out(y1):
            count_event += 1

    p_hat = count_event / n_pairs
    bias = abs(p_hat - 0.5)
    sigma = sqrt(0.25 / n_pairs)

    if verbose:
        print(f"Rondas: {rounds}")
        print(f"Pairs: {n_pairs}")
        print(f"p_hat: {p_hat:.6f}, bias: {bias:.4e}, sigma: {sigma:.4e}")
        print(f"bias/sigma = {bias/sigma:.2f}\n")

    return p_hat, bias, sigma


# =====================================================
# 3. Ejecución de experimentos y graficación
# =====================================================

def run_experiments():
    """
    Ejecuta varias configuraciones de ChaCha reducido,
    guarda los resultados en un CSV y produce una gráfica
    p_hat vs log2(n_pairs) para distintos números de rondas.

    Esta función replica el comportamiento observado en la parte
    experimental del artículo, pero en un entorno reducido.
    """
    key = os.urandom(32)  # clave fija

    experiments = [
        (2, 2**16),
        (2, 2**18),
        (3, 2**16),
        (3, 2**18),
        (3, 2**20),
        (4, 2**18),
        (4, 2**20),
    ]

    results = []

    for rounds, n_pairs in experiments:
        p_hat, bias, sigma = experiment_chacha_dl(
            key=key,
            rounds=rounds,
            n_pairs=n_pairs,
            diff_word_index=14,
            diff_value=0x00000040,
            output_word_indices=(3, 4),
            bit_position=0,
            verbose=True
        )

        results.append({
            "rounds": rounds,
            "n_pairs": n_pairs,
            "log2_n_pairs": log2(n_pairs),
            "p_hat": p_hat,
            "bias": bias,
            "sigma": sigma,
            "bias_over_sigma": bias / sigma
        })

    # Guardar resultados
    csv_filename = "C:/Users/prestamour/Downloads/resultados_chacha_dl.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rounds", "n_pairs", "log2_n_pairs", "p_hat", "bias", "sigma", "bias_over_sigma"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nCSV guardado en: {csv_filename}")

    # Gráfica
    plt.figure()
    rounds_set = sorted(set(r["rounds"] for r in results))

    for r in rounds_set:
        xs = [row["log2_n_pairs"] for row in results if row["rounds"] == r]
        ys = [row["p_hat"] for row in results if row["rounds"] == r]
        plt.plot(xs, ys, marker="o", label=f"{r} rondas")

    plt.axhline(0.5, linestyle="--")
    plt.xlabel(r"$\log_2(\text{n\_pairs})$")
    plt.ylabel("Probabilidad empírica")
    plt.title("ChaCha reducido: probabilidad empírica vs número de pares")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiments()
