from setuptools import find_packages, setup

setup(
    name='deep-map',
    packages=find_packages(include=['deep-map', 'deep-map.*']),
    version='0.0.1',
    description='Deep learning-enhanced morphological profiling',
    author='Edward Ren, Alexander Marx'
)