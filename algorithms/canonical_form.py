# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    tt = tt.copy()
    d = tt.order

    for k in range(d - 1):
        core = tt.cores[k]
        r_k, n_k, r_next = core.shape

        # reshape -> (r_k * n_k, r_next)
        mat = backend.reshape(core, (r_k * n_k, r_next))

        Q, R = backend.qr(mat)

        new_rank = Q.shape[1]

        # новое ядро
        Q_core = backend.reshape(Q, (r_k, n_k, new_rank))
        tt.cores[k] = Q_core

        # обновляем следующее ядро
        next_core = tt.cores[k + 1]
        r_next_old, n_next, r_next2 = next_core.shape

        next_mat = backend.reshape(
            next_core,
            (r_next_old, n_next * r_next2)
        )

        updated = backend.matmul(R, next_mat)

        tt.cores[k + 1] = backend.reshape(
            updated,
            (new_rank, n_next, r_next2)
        )

    return TTTensor(tt.cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    tt = tt.copy()
    d = tt.order

    for k in reversed(range(1, d)):
        core = tt.cores[k]
        r_prev, n_k, r_k = core.shape
        mat = backend.reshape(core, (r_prev, n_k * r_k))
        U, S, Vt = backend.svd(mat, full_matrices=False)
        r_new = S.shape[0]
        new_core = backend.reshape(
            Vt,
            (r_new, n_k, r_k)
        )
        tt.cores[k] = new_core
        US = backend.matmul(U, backend.diag(S))
        prev_core = tt.cores[k - 1]
        r_prev2, n_prev, r_prev_old = prev_core.shape
        prev_mat = backend.reshape(
            prev_core,
            (r_prev2 * n_prev, r_prev_old)
        )
        updated = backend.matmul(prev_mat, US)
        tt.cores[k - 1] = backend.reshape(
            updated,
            (r_prev2, n_prev, r_new)
        )
    return TTTensor(tt.cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть 1D вектором")

    if S.size == 0:
        return 0

    max_sigma = max(abs(x) for x in S.data)

    threshold = max(abs_tol, rel_tol * max_sigma)

    rank = 0
    for sigma in S.data:
        if abs(sigma) > threshold:
            rank += 1

    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if rank == 0:
        return backend.zeros((matrix.shape[0], 0))

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
    if rank == 0:
        return backend.zeros((0, matrix.shape[1]))

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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    result = backend.zeros(matrix.shape)

    for i in range(rank):
        for j in range(matrix.shape[1]):
            result[i, j] = diag_vec[i] * matrix[i, j]

    return result


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    result = backend.zeros((m, n))

    for i in range(m):
        for j in range(n):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result