import operator
import struct
from collections.abc import Sequence
from itertools import accumulate, chain
from math import isclose, sqrt, cos, sin, atan2
from typing import overload, Self, Callable, Any

from vector.vector_dimension_exception import VectorDimensionException


class Vector(Sequence[float]):
    """A class that represents a multidimensional vector of floats."""

    @staticmethod
    def _assert_vector_dimensions_match(  # type: ignore[misc]
            func: Callable[[Self, Sequence[float]], Any]):
        def wrapper(v1: Self, v2: Sequence[float]):
            if len(v1) != len(v2):
                error_message = 'Vectors must have the same dimension.'
                raise VectorDimensionException(error_message)
            return func(v1, v2)
        return wrapper

    def __init__(self, *items: float):
        """
        A constructor that creates a vector from a collection of floats.
        :param items: Cartesian coordinates of the vector
        """
        self._items = tuple(x for x in items)

    def __bool__(self) -> bool:
        """Boolean representation of the vector.
        Vector is considered True if it contains at least one element
        that is non-zero. Numbers are compared with absolute tolerance
        of 1e-9.
        :return: A boolean representation of the vector.
        """
        return any(not isclose(x, 0, abs_tol=1e-9) for x in self)

    @classmethod
    def from_polar(cls, *items: float) -> Self:
        """
        Creates a vector from polar (spherical) coordinates.
        :param items: Spherical coordinates of the vector.
        :return: A newly created vector object.
        """
        if len(items) < 2:
            return cls(*items)
        radius = abs(items[0])
        angles = items[1:]
        sines = (sin(x) for x in angles)
        sine_products = chain([1], accumulate(sines, operator.mul))
        cosines = chain((cos(x) for x in angles), [1])
        coordinates = (radius * product * cosine
                       for product, cosine in zip(sine_products, cosines))
        return cls(*coordinates)

    def get_cartesian_coordinates(self) -> tuple[float, ...]:
        """
        Gets cartesian coordinates of the vector
        :return: A tuple with the cartesian coordinates of the vector.
        """
        return self._items

    def get_polar_coordinates(self) -> tuple[float, ...]:
        """
        Gets polar (spherical) coordinates of the vector.

        If vector is empty, the method will return an empty tuple.

        If vector is 1D, it will return the tuple, containing its only
        cartesian coordinate.

        Otherwise, it will return a tuple with spherical coordinates
        (r, phi1, phi2, ..., phiN) where r >= 0, 0 &lt;= phi1,...,phiN-1 &lt;= pi,
        -pi &lt; phiN &lt;= pi.
        :return: A tuple with the polar (spherical) coordinates of
        the vector.
        """
        if len(self) < 2:
            return self._items[:]
        squares = (x ** 2 for x in self[::-1])
        partial_sums = accumulate(squares, operator.add)
        partial_roots = [sqrt(x) for x in partial_sums]
        partial_roots.reverse()
        partial_roots[-1] = self[-1]
        radius = partial_roots[0]
        angles = (atan2(root, x)
                  for root, x in zip(partial_roots[1:], self[:-1]))
        return tuple(chain([radius], angles))

    @classmethod
    def from_bytes(cls, bytestring: bytes) -> Self:
        """
        Creates a vector from a string of bytes.
        :param bytestring: A byte representation of double-precision
        floating point numbers that will form the cartesian coordinates
        of the new vector.
        :return: A newly-formed vector.
        """
        items = (x[0] for x in struct.iter_unpack('d', bytestring))
        return cls(*items)

    def to_bytes(self) -> bytes:
        """
        Create a byte representation of the elements of the vector.
        :return: A byte representation of double-precision floats that
        represent the cartesian coordinates of the vector.
        """
        return struct.pack(f'{len(self)}d', *self)

    def get_coordinate(
            self,
            index: int,
            default: float | None = None) -> float | None:
        """
        Gets the cartesian coordinate of the vector at the specified index.
        If the coordinate does not exist, it returns a provided default value.
        :param index: A non-negative index to take a coordinate.
        :param default: A default value that should be returned if index is
        not present in the vector.
        :return: The coordinate at the specified position or a default value
        if the coordinate is not found.
        """
        return self[index] if len(self) >= index + 1 and index >= 0 else default

    @property
    def x(self) -> float | None:
        """
        The first coordinate of the vector.
        Equivalent to calling get_coordinate(0)
        :return: The first coordinate of the vector.
        """
        return self.get_coordinate(0)

    @property
    def y(self) -> float | None:
        """
        The second coordinate of the vector.
        Equivalent to calling get_coordinate(1)
        :return: The second coordinate of the vector.
        """
        return self.get_coordinate(1)

    @property
    def z(self) -> float | None:
        """
        The third coordinate of the vector.
        Equivalent to calling get_coordinate(2)
        :return: The third coordinate of the vector.
        """
        return self.get_coordinate(2)

    @property
    def t(self) -> float | None:
        """
        The fourth coordinate of the vector.
        Equivalent to calling get_coordinate(3)
        :return: The fourth coordinate of the vector.
        """
        return self.get_coordinate(3)

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...

    def __getitem__(self, index: int | slice) -> float | Sequence[float]:
        """
        Provides read-only index-based access to the elements of the vector.
        It supports integer indices and slices.
        :param index:
        :return:
        """
        if isinstance(index, slice):
            return type(self)(*self._items[index])
        return self._items[index]

    def __len__(self) -> int:
        """
        Returns the number of elements in the vector.
        :return:
        """
        return len(self._items)

    def __repr__(self) -> str:
        """
        A string representation of the vector. A vector containing the
        elements [0, 1, 2, 3] would be presented as Vector(0, 1, 2, 3).
        :return:
        """
        return f'{type(self).__name__}({self.__join_elements()})'

    def __str__(self) -> str:
        """
        A string representation of the vector that is shown when object is
        converted to string. A vector containing the elements [0, 1, 2, 3]
        would be presented as (0, 1, 2, 3).
        :return:
        """
        return f'({self.__join_elements()})'

    def __format__(self, fmt: str) -> str:
        """
        Called when vector is formatted as a string.
        The format string has the same structure as the float format string
        and provides the same formatting to all the coordinates of the vector.
        It also supports an optional 'polar|' prefix which would result
        in displaying the polar (spherical) coordinates instead of
        cartesian.
        :param fmt: A format string. It supports all the formats of the
        float format string and an optional 'polar|' prefix.
        :return: A formatted string that represents the vector.
        """
        use_polar = fmt.startswith('polar|')
        fmt = fmt.partition('polar|')[2] if use_polar else fmt
        return f'({self.__join_elements(fmt, use_polar=use_polar)})'

    def __join_elements(self, fmt: str = '', use_polar: bool = False) -> str:
        coordinates = self.get_polar_coordinates() if use_polar else self._items
        return ", ".join(f'{x:{fmt}}' for x in coordinates)

    def __eq__(self, other: object) -> bool:
        """
        An equality operator.
        :param other: Object to compare the vector to.
        :return: True if the other object is also a vector, has the same
        number of elements and its elements are equal to the elements of
        the current vector. Otherwise, it returns false. Elements are
        compared with the absolute tolerance of 1e-9.
        """
        if not isinstance(other, Vector):
            return False
        if len(self) != len(other):
            return False
        for item1, item2 in zip(self, other):
            if not isclose(item1, item2, abs_tol=1e-9):
                return False
        return True

    def __ne__(self, other: object) -> bool:
        """
        An inequality operator.
        :param other: Object to compare the vector to.
        :return: The opposite value of the equality operator.
        """
        return not self == other

    def __hash__(self) -> int:
        """
        Hashes the vector.
        :return:
        """
        return hash(self._items)

    def __pos__(self) -> Self:
        """
        A unary '+' operator.
        :return: The same vector.
        """
        return self

    def __neg__(self) -> Self:
        """
        A unary '-' operator.
        :return: A new vector with cartesian coordinates multiplied by -1.
        """
        return self * -1

    @_assert_vector_dimensions_match
    def __add__(self, other: Sequence[float]) -> Self:
        """
        A binary '+' operator. It is possible to add any sequence of floats
        containing the same number of elements as this vector.
        :param other: A sequence representing the second operand.
        :return: A new vector containing the elements that are equal to the
        sums of the corresponding elements of the operands.
        :raises VectorDimensionException: If the operands contain different
        number of elements
        """
        return type(self)(*(item1 + item2 for item1, item2 in zip(self, other)))

    def __radd__(self, other: Sequence[float]) -> Self:
        """
        A method that adds support for the binary '+' operator if the vector
        is the right operand and the left operand is of different type.
        :param other: A sequence representing the second operand.
        :return: A new vector containing the elements that are equal to the
        sums of the corresponding elements of the operands.
        :raises VectorDimensionException: If the operands contain different
        number of elements
        """
        return self + other

    def __sub__(self, other: Self) -> Self:
        """
        A binary '-' operator. It is possible to subtract any
        sequence of floats containing the same number of elements
        as this vector.
        :param other: A sequence representing the second operand.
        :return: A new vector containing the elements that are equal to the
        differences of the corresponding elements of the operands.
        :raises VectorDimensionException: If the operands contain different
        number of elements
        """
        return self + (-other)

    @overload
    def __mul__(self, other: Sequence[float]) -> float: ...

    @overload
    def __mul__(self, other: float) -> Self: ...

    def __mul__(self, other: Sequence[float] | float) -> Self | float:
        """
        A binary '*' operator. If the second operand is a Sequence of floats
        then this is a dot product of the two vectors. If the second
        operand is a float then this float is multiplied by every
        element of the vector and a new vector is produced.
        :param other: A sequence or a floating point number representing
        the second operand.
        :return: If the second argument is a Sequence, then a dot product.
        If the second argument is a floating point number then a new vector
        containing the elements that are equal to the products of the
        corresponding elements of the vector and the second operand.
        :raises VectorDimensionException: If the operands contain different
        number of elements
        """
        if isinstance(other, Sequence):
            return self.dot(other)
        return type(self)(*(item * other for item in self))

    @overload
    def __rmul__(self, other: Sequence[float]) -> float: ...

    @overload
    def __rmul__(self, other: float) -> Self: ...

    def __rmul__(self, other: Sequence[float] | float) -> Self | float:
        """
        A method that adds support for the binary '*' operator if the vector
        is the right operand and the left operand is of different type.
        :param other: A sequence or a floating point number representing
        the second operand.
        :return: If the second argument is a Sequence, then a dot product.
        If the second argument is a floating point number then a new vector
        containing the elements that are equal to the products of the
        corresponding elements of the vector and the second operand.
        :raises VectorDimensionException: If the operands contain different
        number of elements
        """
        return self * other

    @_assert_vector_dimensions_match
    def dot(self, other: Sequence[float]) -> float:
        """
        A dot product of the vectors. The argument can be any Sequence
        with the same number of elements as the vector.
        :param other: A Sequence to multiply the current vector by.
        :return: The dot product of the vectors.
        """
        return sum(item1 * item2 for item1, item2 in zip(self, other))

    def __abs__(self) -> float:
        """
        The absolute value of the vector.
        :return:
        """
        return sqrt(self * self)
