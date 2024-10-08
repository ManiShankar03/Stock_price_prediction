from setuptools import setup

setup(
    name='stock_price_prediction',
    version='0.1.0',
    install_requires=[
        'streamlit==1.18.1',
        'pandas==2.0.3',
        'numpy==1.24.4',
        'tensorflow==2.14.0',
        'joblib==1.3.1',
        'plotly==5.13.1',
        'scikit-learn==1.3.2',
        'setuptools==65.5.0',
    ],
    python_requires='>=3.10',
)
