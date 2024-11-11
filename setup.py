from setuptools import find_packages, setup  # type: ignore
from typing import List

HYPHEN_E_DOT = '-e.'

def get_requirements(file_path: str) -> List[str]:
    '''
    Reads a requirements file and returns a list of dependencies.
    Removes the '-e .' entry if present.
    '''
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]  # Strip newline and whitespace
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    
    return requirements

setup(
    name='machine_learning',
    version='0.0.1',
    author='vaishnavi',
    author_email='sanikakhadage@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.6'
)