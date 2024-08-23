from setuptools import setup

setup(
    name='llm_summarize',  # Replace with your package name
    version='0.1.0',
    packages=["llm_summarize"],
    install_requires=[
        'click',
        'pyyaml',
        'tiktoken',
        'openai>=0.28.0',
        'scikit-learn>=1.3.1',
        'langchain>=0.0.300',
        'spacy>=3.6.1',
        'tiktoken',
        'tenacity',
    ],
    entry_points={
        'console_scripts': [
            'llm-summarize = llm_summarize.main:main',  # Replace with your command and module
        ],
    },
    python_requires='>=3.7',  # Specify the required Python version
)
