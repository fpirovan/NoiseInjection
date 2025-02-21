
#!/usr/bin/env python
import os
import io
import re
import shutil
import sys
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open('README.md').read()

VERSION = find_version('dart', '__init__.py')

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'numba',
    'sklearn',
    'keras==2.0.4',
    'mujoco-py==2.0.2.13',
    'gym==0.17.2'
]

setup(
    # Metadata
    name='dart',
    version=VERSION,

    # Package info
    packages=find_packages(exclude=('examples',)),

    zip_safe=True,
    install_requires=requirements
)