import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="bpush",
    version="0.0.1",
    description="Multi-agent environment for reinforcement learning",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Filippos Christianos",
    url="https://github.com/semitable/boulder-push",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "gym>=0.26.2",
        "pettingzoo>=1.21.0",
        "pyglet",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
