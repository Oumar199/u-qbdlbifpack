from setuptools import setup

setup(
    name="translate_package",
    version="0.0.1",
    author="Oumar Kane",
    author_email="oumar.kane@univ-thies.sn",
    description="Contain functions and classes to efficiently train a sequence to sequence to translate between two languages.",
    install_requires=[
        "accelerate==0.21.0",
        # "torch==2.0.0+cu117",
        "spacy",
        "nltk",
        "gensim",
        "furo",
        "streamlit",
        "tensorboard",
        "evaluate",
        "tokenizers==0.13.3",
        "transformers==4.29.2",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "sacrebleu",
        "nlpaug",
        "wandb",
        "pytorch-lightning==1.9.4",
        "selenium",
        "sentencepiece",
        "peft"
    ],
)
