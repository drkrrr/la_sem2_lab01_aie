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
    cores = []
    current = None
    for k in range(tt.order - 1):
        core = tt.cores[k].copy()
        r_left, n_k, r_right = core.shape
        if current is not None:
            new_data = []
            for i in range(current.shape[0]):
                for j in range(n_k):
                    for p in range(r_right):
                        val = 0.0
                        for s in range(r_left):
                            val += current[i, s] * core[s, j, p]
                        new_data.append(val)
            core = DenseTensor((current.shape[0], n_k, r_right), data=new_data)
            r_left = current.shape[0]
        matrix = core.reshape((r_left * n_k, r_right))
        m, n = matrix.shape
        if m < n:
            U, S, Vt = backend.svd(matrix)
            rank = min(m, n)

            Q = _truncate_columns(U, rank)
            R = _multiply_diag_matrix(
                S,
                _truncate_rows(Vt, rank),
                rank
            )
        else:
            Q, R = backend.qr(matrix)
        new_core = Q.reshape((r_left, n_k, Q.shape[1]))
        cores.append(new_core)
        current = R
    last_core = tt.cores[-1].copy()
    if current is not None:
        r_left, n_last, r_right = last_core.shape
        new_data = []
        for i in range(current.shape[0]):
            for j in range(n_last):
                for p in range(r_right):
                    val = 0.0
                    for s in range(r_left):
                        val += current[i, s] * last_core[s, j, p]
                    new_data.append(val)
        last_core = DenseTensor(
            (current.shape[0], n_last, r_right),
            data=new_data
        )
    cores.append(last_core)
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    d = tt.order
    cores = [None] * d
    current = None
    for k in range(d - 1, 0, -1):
        core = tt.cores[k].copy()
        if current is not None:
            new_data = []
            for i in range(core.shape[0]):
                for j in range(core.shape[1]):
                    for p in range(current.shape[1]):
                        val = 0.0
                        for s in range(core.shape[2]):
                            val += core[i, j, s] * current[s, p]
                        new_data.append(val)
            core = DenseTensor(
                (core.shape[0], core.shape[1], current.shape[1]),
                data=new_data
            )
        r_left, n_k, r_right = core.shape
        matrix = core.reshape((r_left, n_k * r_right))
        matrix_t = backend.transpose(matrix)
        m, n = matrix_t.shape
        if m < n:
            U, S, Vt = backend.svd(matrix_t)
            rank = min(m, n)
            Q_t = _truncate_columns(U, rank)
            R_t = _multiply_diag_matrix(
                S,
                _truncate_rows(Vt, rank),
                rank
            )
        else:
            Q_t, R_t = backend.qr(matrix_t)
        Q = backend.transpose(Q_t)
        R = backend.transpose(R_t)
        new_rank = Q.shape[0]
        cores[k] = Q.reshape((new_rank, n_k, r_right))
        current = R
    first_core = tt.cores[0].copy()
    if current is not None:
        new_data = []
        for i in range(first_core.shape[0]):
            for j in range(first_core.shape[1]):
                for p in range(current.shape[1]):
                    val = 0.0
                    for s in range(first_core.shape[2]):
                        val += first_core[i, j, s] * current[s, p]
                    new_data.append(val)
        first_core = DenseTensor(
            (first_core.shape[0], first_core.shape[1], current.shape[1]),
            data=new_data
        )
    cores[0] = first_core
    for i in range(d):
        if cores[i] is None:
            cores[i] = tt.cores[i].copy()
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
    if S.ndim != 1:
        raise ValueError('S должен быть 1D')
    if S.size == 0:
        return 0
    threshold = max(abs_tol, rel_tol * max(abs(x) for x in S.data))
    return sum(1 for value in S.data if abs(value) > threshold)


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
    rows, cols = matrix.shape
    if matrix.ndim != 2 or rank < 0 or rank > cols:
        raise ValueError('некорр rank')
    result = backend.zeros((rows, rank))
    for i in range(rows):
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
    rows, cols = matrix.shape
    if matrix.ndim != 2 or rank < 0 or rank > rows:
        raise ValueError('некорр rank')
    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
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
    if vector.ndim != 1 or rank < 0 or rank > vector.shape[0]:
        raise ValueError('некорр rank')
    return DenseTensor((rank,), data=vector.data[:rank])


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
    cols = matrix.shape[1]
    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i, j] = diag_vec.data[i] * matrix[i, j]
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
    rows, cols = matrix.shape
    if diag_vec.shape[0] != cols:
        raise ValueError
    result = backend.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix[i, j] * diag_vec.data[j]
    return result