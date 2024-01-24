from collections.abc import Sequence
from math import pi, sqrt, isclose

import pytest

from vector.vector import Vector
from vector.vector_dimension_exception import VectorDimensionException


class TestVector:
    @pytest.mark.parametrize(
        'expected,test_input', [
            ((0, 0, 0), (0, pi, pi / 2)),
            ((5, 0, 0), (5, 0, 0)),
            ((0, 5, 0), (5, pi / 2, 0)),
            ((-5, 0, 0), (5, pi, 0)),
            ((0, -5, 0), (5, 3 * pi / 2, 0)),
            ((5, 0, 0), (5, 0, pi / -2)),
            ((0, 0, -5), (5, pi / 2, pi / -2)),
            ((-5, 0, 0), (5, pi, pi / -2)),
            ((0, 0, 5), (5, 3 * pi / 2, pi / -2)),
            ((5, 0, 0), (5, 0, pi * -1)),
            ((0, -5, 0), (5, pi / 2, pi * -1)),
            ((-5, 0, 0), (5, pi, pi * -1)),
            ((0, 5, 0), (5, 3 * pi / 2, pi * -1)),
            ((0, 0, 5), (5, pi / 2, 3 * pi / -2)),
            ((-5, 0, 0), (5, pi, 3 * pi / -2)),
            ((0, 0, -5), (5, 3 * pi / 2, 3 * pi / -2)),
            ((sqrt(3), sqrt(3) / 2, -0.5), (2, pi / 6, pi / -6)),
            ((sqrt(3), sqrt(3) / 2, -0.5), (-2, 13 * pi / 6, 13 * pi / -6)),
            ((sqrt(3) * -1, sqrt(3) / 2, -0.5), (2, 5 * pi / 6, pi / -6)),
            ((sqrt(3) * -1, sqrt(3) / -2, 0.5), (2, 7 * pi / 6, pi / -6)),
            ((sqrt(3), sqrt(3) / -2, 0.5), (2, 11 * pi / 6, pi / -6)),
            ((sqrt(3), sqrt(3) / -2, -0.5), (2, pi / 6, 5 * pi / -6)),
            ((sqrt(3) * -1, sqrt(3) / -2, -0.5), (2, 5 * pi / 6, 5 * pi / -6)),
            ((sqrt(3), sqrt(3) / 2, 0.5), (2, 11 * pi / 6, 5 * pi / -6)),
            ((sqrt(3), sqrt(3) / -2, 0.5), (2, pi / 6, 7 * pi / -6)),
            ((sqrt(3) * -1, sqrt(3) / -2, 0.5), (2, 5 * pi / 6, 7 * pi / -6)),
            ((sqrt(3) * -1, sqrt(3) / 2, -0.5), (2, 7 * pi / 6, 7 * pi / -6)),
            ((sqrt(3), sqrt(3) / 2, -0.5), (2, 11 * pi / 6, 7 * pi / -6)),
            ((sqrt(3), sqrt(3) / 2, 0.5), (2, pi / 6, 11 * pi / -6)),
            ((sqrt(3) * -1, sqrt(3) / 2, 0.5), (2, 5 * pi / 6, 11 * pi / -6)),
            ((sqrt(3), sqrt(3) / -2, -0.5), (2, 11 * pi / 6, 11 * pi / -6)),
        ]
    )
    def test_from_polar(
            self,
            expected: tuple[float, ...],
            test_input: tuple[float, ...]):
        assert Vector(*expected) == Vector.from_polar(*test_input)

    @pytest.mark.parametrize(
        'expected,test_input',
        [
            ((0, 1, 2, 3),
             b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?'
             b'\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@')
        ]
    )
    def test_from_bytes(self, expected: tuple[float, ...], test_input: bytes):
        assert Vector(*expected) == Vector.from_bytes(test_input)

    @pytest.mark.parametrize(
        'expected,test_input',
        [
            (b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?'
             b'\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@',
             (0, 1, 2, 3))
        ]
    )
    def test_to_bytes(self, expected: bytes, test_input: tuple[float, ...]):
        assert expected == Vector(*test_input).to_bytes()

    @pytest.mark.parametrize('expected', [
        ((0, 1, 2, 3),)
    ])
    def test_get_cartesian_coordinates(self, expected: tuple[float, ...]):
        assert expected == Vector(*expected).get_cartesian_coordinates()

    @pytest.mark.parametrize(
        'expected,test_input', [
            ((0, 0, 0), (0, 0, 0)),
            ((5, 0, 0), (5, 0, 0)),
            ((5, pi / 2, 0), (0, 5, 0)),
            ((5, pi, 0), (-5, 0, 0)),
            ((5, pi / 2, pi), (0, -5, 0)),
            ((5, pi / 2, pi / -2), (0, 0, -5)),
            ((5, pi / 2, pi / 2), (0, 0, 5)),
            ((2, pi / 6, pi / -6), (sqrt(3), sqrt(3) / 2, -0.5)),
            ((2, 5 * pi / 6, pi / -6), (sqrt(3) * -1, sqrt(3) / 2, -0.5)),
            ((2, 5 * pi / 6, 5 * pi / 6), (sqrt(3) * -1, sqrt(3) / -2, 0.5)),
            ((2, pi / 6, 5 * pi / 6), (sqrt(3), sqrt(3) / -2, 0.5)),
            ((2, pi / 6, 5 * pi / -6), (sqrt(3), sqrt(3) / -2, -0.5)),
            ((2, 5 * pi / 6, 5 * pi / -6), (sqrt(3) * -1, sqrt(3) / -2, -0.5)),
            ((2, pi / 6, pi / 6), (sqrt(3), sqrt(3) / 2, 0.5)),
            ((2, 5 * pi / 6, pi / 6), (sqrt(3) * -1, sqrt(3) / 2, 0.5)),
        ]
    )
    def test_get_polar_coordinates(
            self,
            expected: tuple[float, ...],
            test_input: tuple[float, ...]):
        test_vector = Vector(*test_input)
        for ex, test in zip(expected, test_vector.get_polar_coordinates()):
            assert isclose(ex, test, abs_tol=1e-9)

    @pytest.mark.parametrize(
        'expected,test_input,coordinate', [
            (3, (0, 1, 2, 3), 3),
            (None, (0, 1, 2), 3),
            (None, (0, 1, 2), -6),
        ]
    )
    def test_get_coordinate(
            self,
            expected: float | None,
            test_input: tuple[float, ...],
            coordinate: int):
        assert expected == Vector(*test_input).get_coordinate(coordinate)

    @pytest.mark.parametrize(
        'expected,test_input', [
            (0, (0, 1, 2, 3)),
            (None, []),
        ]
    )
    def test_x(self, expected: float | None, test_input: tuple[float, ...]):
        assert expected == Vector(*test_input).x

    @pytest.mark.parametrize(
        'expected,test_input',
        [
            (1, (0, 1, 2, 3)),
            (None, [0]),
        ]
    )
    def test_y(self, expected: float | None, test_input: tuple[float, ...]):
        assert expected == Vector(*test_input).y

    @pytest.mark.parametrize(
        'expected,test_input', [
            (2, (0, 1, 2, 3)),
            (None, (0, 1)),
        ]
    )
    def test_z(self, expected: float | None, test_input: tuple[float, ...]):
        assert expected == Vector(*test_input).z

    @pytest.mark.parametrize(
        'expected,test_input', [
            (3, (0, 1, 2, 3)),
            (None, (0, 1, 2)),
        ])
    def test_t(self, expected: float | None, test_input: tuple[float, ...]):
        assert expected == Vector(*test_input).t

    @pytest.mark.parametrize(
        'expected,test_input', [
            (-14, ((0, 1, 2, 3), (4, -5, 6, -7))),
        ]
    )
    def test_dot(
            self,
            expected: float,
            test_input: tuple[tuple[float, ...], tuple[float, ...]]):
        assert expected == Vector(*test_input[0]).dot(Vector(*test_input[1]))

    @pytest.mark.parametrize(
        'test_input', [
            ((0, 1, 2, 3), (4, -5, 6)),
        ]
    )
    def test_dot_exceptions(
            self,
            test_input: tuple[tuple[float, ...], tuple[float, ...]]):
        with pytest.raises(VectorDimensionException):
            Vector(*test_input[0]).dot(Vector(*test_input[1]))

    @pytest.mark.parametrize(
        'expected,test_input', [
            (sqrt(14), (0, 1, -2, 3)),
            (0, ()),
        ]
    )
    def test_abs(self, expected: float, test_input: tuple[float, ...]):
        assert expected == abs(Vector(*test_input))

    @pytest.mark.parametrize(
        'expected,test_input,index', [
            (0, (0, 1, 2, 3), 0),
        ]
    )
    def test_getitem_index(
            self,
            expected: float,
            test_input: tuple[float],
            index: int):
        assert expected == Vector(*test_input)[index]

    @pytest.mark.parametrize(
        'expected,test_input,index', [
            ((0, 1), (0, 1, 2, 3), (0, 2)),
        ]
    )
    def test_getitem_slice(
            self,
            expected: tuple[float, ...],
            test_input: tuple[float],
            index: tuple[int, int]):
        assert Vector(*expected) == Vector(*test_input)[index[0]:index[1]]

    @pytest.mark.parametrize(
        'expected,test_input', [
            (4, (0, 1, 2, 3)),
        ]
    )
    def test_len(self, expected: int, test_input: tuple[float]):
        assert expected == len(Vector(*test_input))

    @pytest.mark.parametrize(
        'test_input', [
            ((0, 1, 2, 3),),
        ]
    )
    def test_hash(self, test_input: tuple[float, ...]):
        assert hash(test_input) == hash(Vector(*test_input))

    @pytest.mark.parametrize(
        'expected,test_input', [
            ('(0, 1, 2, 3)', (0, 1, 2, 3)),
            ('()', ()),
        ]
    )
    def test_str(self, expected: str, test_input: tuple[float, ...]):
        assert expected == str(Vector(*test_input))

    @pytest.mark.parametrize(
        'expected,test_input', [
            ('Vector(0, 1, 2, 3)', (0, 1, 2, 3)),
            ('Vector()', ()),
        ]
    )
    def test_repr(self, expected: str, test_input: tuple[float, ...]):
        assert expected == repr(Vector(*test_input))

    @pytest.mark.parametrize(
        'expected,test_input,fmt', [
            ('(0, 1, 2, 3)', (0, 1, 2, 3), ''),
            ('(0.00, 1.00, 2.00, 3.00)', (0, 1, 2, 3), '.2f'),
            ('(3.74, 1.57, 1.30, 0.98)', (0, 1, 2, 3), 'polar|.2f'),
            ('()', (), ''),
            ('()', (), '.2f'),
            ('()', (), 'polar|.2f'),
        ]
    )
    def test_format(
            self,
            expected: str,
            test_input: tuple[float, ...],
            fmt: str):
        assert expected == f'{Vector(*test_input):{fmt}}'

    @pytest.mark.parametrize(
        'expected,test_input', [
            (True, (0, 1, 2, 3)),
            (True, (0, 0, 0, 0.001)),
            (False, (0, 0, 0, 1e-16)),
            (False, (0, 0, 0, 0)),
            (False, ()),
        ]
    )
    def test_bool(self, expected: bool, test_input: tuple[float, ...]):
        assert expected == bool(Vector(*test_input))

    @pytest.mark.parametrize(
        'expected,test_input', [
            (True, ((0, 1, 2, 3), (0, 1, 2, 3))),
            (False, ((0, 1, 2, 3), (0, 1, 3, 3))),
            (False, ((0, 1, 2, 3), (0, 1, 2))),
        ]
    )
    def test_equals(
            self,
            expected: bool,
            test_input: tuple[tuple[float, ...], tuple[float, ...]]):
        vector1 = Vector(*test_input[0])
        vector2 = Vector(*test_input[1])
        assert expected == (vector1 == vector2)

    @pytest.mark.parametrize(
        'expected,test_input', [
            (False, ((0, 1, 2, 3), (0, 1, 2, 3))),
            (True, ((0, 1, 2, 3), (0, 1, 3, 3))),
            (True, ((0, 1, 2, 3), (0, 1, 2))),
        ]
    )
    def test_not_equals(
            self,
            expected: bool,
            test_input: tuple[tuple[float, ...], tuple[float, ...]]):
        vector1 = Vector(*test_input[0])
        vector2 = Vector(*test_input[1])
        assert expected == (vector1 != vector2)

    @pytest.mark.parametrize(
        'expected,test_input', [
            ((4, -4, 8, -4), ((0, 1, 2, 3), (4, -5, 6, -7))),
            ((4, -4, 8, -4), ((0, 1, 2, 3), Vector(4, -5, 6, -7))),
            ((4, -4, 8, -4), ((0, 1, 2, 3), [4, -5, 6, -7])),
        ]
    )
    def test_add(
            self,
            expected: tuple[float, ...],
            test_input: tuple[Sequence[float], Sequence[float]]):
        expected_vector = Vector(*expected)
        vector1 = Vector(*test_input[0])
        assert expected_vector\
               == (vector1 + test_input[1])\
               == (test_input[1] + vector1)

    @pytest.mark.parametrize(
        'test_input', [
            ((0, 1, 2, 3), (4, -5, 6)),
        ]
    )
    def test_add_exceptions(
            self,
            test_input: tuple[tuple[float, ...], tuple[float, ...]]):
        with pytest.raises(VectorDimensionException):
            Vector(*test_input[0]) + test_input[1]
        with pytest.raises(VectorDimensionException):
            test_input[1] + Vector(*test_input[0])

    @pytest.mark.parametrize(
        'expected,test_input', [
            (-14, ((0, 1, 2, 3), (4, -5, 6, -7))),
            (-14, ((0, 1, 2, 3), Vector(4, -5, 6, -7))),
            (-14, ((0, 1, 2, 3), [4, -5, 6, -7])),
        ]
    )
    def test_mul_sequence(
            self,
            expected: float,
            test_input: tuple[tuple[float, ...], Sequence[float]]):
        vector1 = Vector(*test_input[0])
        assert expected == vector1 * test_input[1] == test_input[1] * vector1

    @pytest.mark.parametrize(
        'expected,test_input', [
            ((0, -2.5, -5, -7.5), ((0, 1, 2, 3), -2.5)),
        ]
    )
    def test_mul_float(
            self,
            expected: tuple[float, ...],
            test_input: tuple[tuple[float, ...], float]):
        expected_vector = Vector(*expected)
        vector1 = Vector(*test_input[0])
        assert expected_vector\
               == vector1 * test_input[1]\
               == test_input[1] * vector1

    @pytest.mark.parametrize(
        'test_input', [
            ((0, 1, 2, 3), (4, -5, 6)),
        ]
    )
    def test_mul_exceptions(
            self,
            test_input: tuple[tuple[float, ...], tuple[float, ...]]):
        with pytest.raises(VectorDimensionException):
            Vector(*test_input[0]) * test_input[1]
        with pytest.raises(VectorDimensionException):
            test_input[1] * Vector(*test_input[0])

    @pytest.mark.parametrize(
        'expected,test_input', [
            ((0, -1, 2, -3), (0, -1, 2, -3)),
        ]
    )
    def test_pos(
            self,
            expected: tuple[float, ...],
            test_input: tuple[float, ...]):
        assert Vector(*expected) == +Vector(*test_input)

    @pytest.mark.parametrize(
        'expected,test_input', [
            ((0, 1, -2, 3), (0, -1, 2, -3)),
        ]
    )
    def test_neg(
            self,
            expected: tuple[float, ...],
            test_input: tuple[float, ...]):
        assert Vector(*expected) == -Vector(*test_input)
