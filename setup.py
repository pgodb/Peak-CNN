#!/usr/bin/env python3

# Author: Florian Philipp
# Copyright: German Aerospace Center 2020


import setuptools


with open("README.md", 'r') as readme:
    long_description = readme.read()


setuptools.setup(
    name="PeakCNN",
    version="0.1.0",
    author="Philipp Godbersen",
    author_email="philipp.godbersen@dlr.de",
    description="Peak detection on measurement images via CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgodb/Peak-CNN",
    packages=('PeakCNN', ),
    license="MIT",
    classifiers=("Development Status :: 3 - Alpha",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License", 
                 "Operating System :: OS Independent",
                 "Programming Language :: Python",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Software Development :: Libraries :: Python Modules",
                 ),
    )

