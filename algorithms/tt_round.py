# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = tt.order
    tt = right_canonicalize(tt, backend)
    cores = tt.cores.copy()

    norm_tt = backend.norm(tt.full())
    delta = eps * norm_tt / math.sqrt(d - 1) if d > 1 else 0.0

    for k in range(d - 1):
        core = cores[k]
        r_k, n_k, r_next = core.shape
        mat = backend.reshape(core, (r_k * n_k, r_next))
        U, S, Vt = backend.svd(mat, full_matrices=False)
        r = _compute_rank(S, delta, max_rank)
        U = _truncate_columns(U, r, backend)
        S = _truncate_vector(S, r, backend)
        Vt = _truncate_rows(Vt, r, backend)

        cores[k] = backend.reshape(U, (r_k, n_k, r))

        SV = _multiply_diag_matrix(S, Vt, r, backend)

        next_core = cores[k + 1]
        r_next_old, n_next, r_next2 = next_core.shape

        next_mat = backend.reshape(
            next_core,
            (r_next_old, n_next * r_next2)
        )

        updated = backend.matmul(SV, next_mat)

        cores[k + 1] = backend.reshape(
            updated,
            (r, n_next, r_next2)
        )

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимаъьно допустимый ранг (None = без ограничения)
    """
    r = 0
    for sigma in S.data:
        if sigma > delta:
            r += 1

    if max_rank is not None:
        r = min(r, max_rank)

    return max(r, 1)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    m, n = matrix.shape
    result = backend.zeros((m, rank))

    for i in range(m):
        for j in range(rank):
            result[i, j] = matrix[i, j]

    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    k, n = matrix.shape
    result = backend.zeros((rank, n))

    for i in range(rank):
        for j in range(n):
            result[i, j] = matrix[i, j]

    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    result = backend.zeros((rank,))
    for i in range(rank):
        result[i] = vector[i]

    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    result = backend.zeros(matrix.shape)

    for i in range(rank):
        for j in range(matrix.shape[1]):
            result[i, j] = diag_vec[i] * matrix[i, j]

    return result