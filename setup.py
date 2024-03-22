from setuptools import setup, find_packages

setup(
    name='chinese_word_segmenter',
    version='0.1',
    packages=find_packages(),
    license='MIT License',
    description='A BERT-based Chinese Word Segmentation Model Specific to Traditional Chinese (zh_TW)',
    long_description=open('README.md').read(),
    install_requires=[
        "transformers", 
        "simpletransformers",
        "pandas",
        "requests",
        # List all project dependencies here.
        # For example: 'requests >= 2.19.1'
    ],
    url='https://github.com/hhhuang/ChineseWordSegmenter',
    author='Hen-Hsen Huang',
    author_email='huangs@gmail.com',
    classifiers=[
        # Classifiers help users find your project by categorizing it.
        
        # For a list of valid classifiers, see https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='Chinese NLP word segmentation tokenize',  # Keywords that define your package best
    entry_points={
        'console_scripts': [
            # Entry points create executable commands by pointing
            # to a function in your package.
            # For example: 'your_command = your_package.module:function'
        ],
    },
    # Include package data files from MANIFEST.in
    include_package_data=False,
    # Other arguments here...
)
