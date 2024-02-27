from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class Base:
    pass


@dataclass
class A:
    a: int


@dataclass
class B:
    b: int


a = A(a=2)
