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
    cores = [backend.copy(c) for c in tt.cores]
    d = tt.order

    for k in range(d - 1):
        r_l, n_k, r_r = cores[k].shape
        rows = r_l * n_k

        C = backend.reshape(cores[k], (rows, r_r))

        if rows >= r_r:
            Q, R = backend.qr(C)
            _, r_new = Q.shape
            cores[k] = backend.reshape(Q, (r_l, n_k, r_new))
        else:
            CT = backend.transpose(C)
            Q, R = backend.qr(CT)
            _, r_new = Q.shape
            RT = backend.transpose(R)
            cores[k] = backend.reshape(RT, (r_l, n_k, r_new))
            R = backend.transpose(Q)

        r_l2, n_k1, r_r2 = cores[k + 1].shape
        C_next = backend.reshape(cores[k + 1], (r_l2, n_k1 * r_r2))
        C_next = backend.matmul(R, C_next)
        cores[k + 1] = backend.reshape(C_next, (r_new, n_k1, r_r2))

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [backend.copy(c) for c in tt.cores]
    d = tt.order
    for k in range(d - 1, 0, -1):
        r_l, n_k, r_r = cores[k].shape
        cols = n_k * r_r

        C = backend.reshape(cores[k], (r_l, cols))

        if cols >= r_l:
            CT = backend.transpose(C)
            Q, R = backend.qr(CT)
            _, r_new = Q.shape
            QT = backend.transpose(Q)
            cores[k] = backend.reshape(QT, (r_new, n_k, r_r))
            L = backend.transpose(R)
        else:
            Q, R = backend.qr(C)
            _, r_new = Q.shape
            cores[k] = backend.reshape(R, (r_new, n_k, r_r))
            L = Q

        r_l2, n_km1, _ = cores[k - 1].shape
        C_prev = backend.reshape(cores[k - 1], (r_l2 * n_km1, r_l))
        C_prev = backend.matmul(C_prev, L)
        cores[k - 1] = backend.reshape(C_prev, (r_l2, n_km1, r_new))

    return TTTensor(cores)


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
    if S.size == 0:
        return 0
    sigma_max = abs(S.data[0])
    threshold = max(abs_tol, rel_tol * sigma_max)
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
    m, n = matrix.shape
    rank = min(rank, n)
    result = backend.zeros((m, rank))
    for i in range(m):
        for j in range(rank):
            v = backend.get_element(matrix, (i, j))
            backend.set_element(result, (i, j), v)
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
    rank = min(rank, k)
    result = backend.zeros((rank, n))
    for i in range(rank):
        for j in range(n):
            v = backend.get_element(matrix, (i, j))
            backend.set_element(result, (i, j), v)
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
    rank = min(rank, vector.shape[0])
    result = backend.zeros((rank,))
    for i in range(rank):
        v = backend.get_element(vector, (i,))
        backend.set_element(result, (i,), v)
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
    D = backend.diag(diag_vec)
    return backend.matmul(D, matrix)


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
    D = backend.diag(diag_vec)
    return backend.matmul(matrix, D)