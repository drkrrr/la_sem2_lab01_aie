# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations
import random
from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if len(cores) == 0:
            raise ValueError("Список ядер не может быть пустым")
        self.cores = cores
        self.order = len(cores)
        shape = []
        ranks = []
        for i, core in enumerate(cores):
            if core.ndim != 3:
                raise ValueError("Каждое ядро должно быть 3D-тензором")
            r_left, n, r_right = core.shape
            if i == 0 and r_left != 1:
                raise ValueError("Первый TT-ранг должен быть 1")
            if i == len(cores) - 1 and r_right != 1:
                raise ValueError("Последний TT-ранг должен быть 1")
            if i > 0:
                prev_r = cores[i - 1].shape[2]
                if r_left != prev_r:
                    raise ValueError("Несогласованные TT-ранги")
            shape.append(n)
            ranks.append(r_left)
        ranks.append(cores[-1].shape[2])
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shape = validate_shape(shape)
        d = len(shape)
        if seed is not None:
            random.seed(seed)
        if len(ranks) == d - 1:
            ranks = (1,) + tuple(ranks) + (1,)
        else:
            ranks = tuple(ranks)

        if len(ranks) != d + 1:
            raise ValueError("Неверная длина ranks")
        cores = []
        for k in range(d):
            r_left = ranks[k]
            n = shape[k]
            r_right = ranks[k + 1]
            data = [
                random.uniform(-1.0, 1.0)
                for _ in range(r_left * n * r_right)
            ]
            cores.append(
                DenseTensor((r_left, n, r_right), data=data)
            )
        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise ValueError("Неверная длина индексов")

            # начинаем с вектора размерности r0=1
        vec = [1.0]

        for k in range(self.order):

            core = self.cores[k]
            i_k = indices[k]

            r_left, n, r_right = core.shape

            new_vec = [0.0] * r_right

            for alpha in range(r_left):
                for beta in range(r_right):
                    new_vec[beta] += (
                            vec[alpha] *
                            core[alpha, i_k, beta]
                    )
            vec = new_vec
        return vec[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        full_tensor = DenseTensor.zeros(self.shape)
        for flat_index in range(full_tensor.size):
            multi = full_tensor.shape
            indices = []
            remainder = flat_index
            strides = full_tensor.strides
            for stride, dim in zip(strides, full_tensor.shape):
                val = remainder // stride
                remainder %= stride
                indices.append(val)
            value = self.get_element(indices)
            full_tensor.data[flat_index] = value
        return full_tensor

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        return full_size / tt_size

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = []
        lines.append("TTTensor:")
        lines.append(f"order: {self.order}")
        lines.append(f"shape: {self.shape}")
        lines.append(f"ranks: {self.ranks}")
        lines.append(f"core shapes: {self.core_sizes()}")
        lines.append(f"storage: {self.total_storage()} elements")

        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()