from setuptools import setup, Extension, find_packages

setup(
    name='mykmeanssp',
    version='1.0',
    description='kmeans algo in c',
    packages=find_packages(),
    ext_modules=[
        Extension('mykmeanssp', ['kmeans.c'])
    ]
)