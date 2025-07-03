from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="gitlab-gcr-gke-project",
    version="0.1",
    author="Jatin",
    packages=find_packages(),
    install_requires = requirements,
)