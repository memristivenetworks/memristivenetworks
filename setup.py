# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""memristivenetworks: Library for simulating memristive networks

simulates the functioning of memristive Willshaw networks and memristive
perceptrons

"""

from setuptools import setup

description = 'simulates the functioning of memristive Willshaw networks '
description += 'and memristive perceptrons.'


setup(name='memristivenetworks',
      version='0.0.1',
      description='Library for simulating memristive networks',
      long_description=description,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      url='https://github.com/danieljosesilva/memristivenetworks',
      author='Daniel Silva',
      author_email='djsilva99@gmail.com',
      license='MIT',
      packages=['memristivenetworks'],
      install_requires=[
          'numpy',
          'matplotlib'
      ],
      include_package_data=True,
      zip_safe=False)
