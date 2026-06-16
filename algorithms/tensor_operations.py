# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("Формы TT-тензоров не совпадают")

    d = tt1.order
    new_cores = []

    for k in range(d):

        G1 = tt1.cores[k]
        G2 = tt2.cores[k]

        r1_l, n, r1_r = G1.shape
        r2_l, _, r2_r = G2.shape
        if k == 0:
            new_core = backend.zeros((1, n, r1_r + r2_r))
            for j in range(n):
                for b in range(r1_r):
                    new_core[0, j, b] = G1[0, j, b]
                for b in range(r2_r):
                    new_core[0, j, r1_r + b] = G2[0, j, b]

        elif k == d - 1:
            new_core = backend.zeros((r1_l + r2_l, n, 1))
            for a in range(r1_l):
                for j in range(n):
                    new_core[a, j, 0] = G1[a, j, 0]
            for a in range(r2_l):
                for j in range(n):
                    new_core[r1_l + a, j, 0] = G2[a, j, 0]

        else:
            new_core = backend.zeros(
                (r1_l + r2_l, n, r1_r + r2_r)
            )

            for a in range(r1_l):
                for j in range(n):
                    for b in range(r1_r):
                        new_core[a, j, b] = G1[a, j, b]

            for a in range(r2_l):
                for j in range(n):
                    for b in range(r2_r):
                        new_core[r1_l + a, j, r1_r + b] = G2[a, j, b]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    cores = tt.cores.copy()
    cores[0] = backend.scale(cores[0], alpha)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("Формы TT-тензоров не совпадают")

    d = tt1.order
    new_cores = []

    for k in range(d):
        G1 = tt1.cores[k]
        G2 = tt2.cores[k]
        r1_l, n, r1_r = G1.shape
        r2_l, _, r2_r = G2.shape
        new_core = backend.zeros(
            (r1_l * r2_l, n, r1_r * r2_r)
        )
        for a1 in range(r1_l):
            for a2 in range(r2_l):
                for j in range(n):
                    for b1 in range(r1_r):
                        for b2 in range(r2_r):
                            new_core[
                                a1 * r2_l + a2,
                                j,
                                b1 * r2_r + b2
                            ] = (
                                    G1[a1, j, b1] *
                                    G2[a2, j, b2]
                            )
        new_cores.append(new_core)
    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("Формы TT-тензоров не совпадают")

    d = tt1.order

    # начальная матрица 1x1
    C = [[1.0]]

    for k in range(d):

        G1 = tt1.cores[k]
        G2 = tt2.cores[k]

        r1_l, n, r1_r = G1.shape
        r2_l, _, r2_r = G2.shape

        new_C = [
            [0.0 for _ in range(r1_r * r2_r)]
            for _ in range(1)
        ]

        for a1 in range(r1_l):
            for a2 in range(r2_l):
                coeff = C[0][a1 * r2_l + a2] if k > 0 else 1.0

                for j in range(n):
                    for b1 in range(r1_r):
                        for b2 in range(r2_r):
                            idx = b1 * r2_r + b2
                            new_C[0][idx] += (
                                    coeff *
                                    G1[a1, j, b1] *
                                    G2[a2, j, b2]
                            )

        C = new_C

    return C[0][0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    diff = tt_add(
        tt1,
        tt_scalar_mul(tt2, -1, backend),
        backend
    )

    return tt_norm(diff, backend)
