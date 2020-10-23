import sys

from setuptools import setup, find_packages
from os import path
from io import open


here = path.abspath(path.dirname(__file__))

sys.path.insert(0, path.join(here, 'metax'))
from version import VERSION

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='metax',
    version=VERSION,
    description='A collection of extensions for meta-learning in JAX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author='Tristan Deleu',
    author_email='tristan.deleu@gmail.com',
    url='https://github.com/tristandeleu/metax',
    keywords=['meta-learning', 'jax', 'few-shot', 'few-shot learning'],
    packages=find_packages(exclude=['data', 'tests', 'examples']),
    install_requires=['numpy', 'jax', 'jaxlib'],
)