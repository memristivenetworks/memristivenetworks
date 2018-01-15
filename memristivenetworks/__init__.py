# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""memristivenetworks.

Created for chapter Associative networks and perceptron based on memristors:
fundamentals and algorithmic implementation of the book Handbook of Memristor
Networks, Springer 2018.

The package relies on 2 main classes:

#########
perceptron
#########

See Appendix 2 - Python code for memristive-based
perceptron.

#########
willshaw
#########

See Appendix 1 - Python code for memristive-based
Willshaw network.

"""

from willshaw import Willshaw
from perceptron import Perceptron

__all__ = [Willshaw, Perceptron]
