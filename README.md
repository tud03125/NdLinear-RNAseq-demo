<p align="center">
  <img src="ensemble_logo.jpg" alt="Logo" width="400">
  <br /> <br / >
</p>

<!--
  README.md for NdLinear-RNAseq-demo  
  Demonstrates application of NdLinear to human liver RNA-seq (GSE126848).
-->

## Summary

This repository demonstrates how to apply the [NdLinear](https://github.com/ensemble-core/NdLinear) library to bulk RNA-seq data (GSE126848), covering data loading, variance filtering, ElasticNet feature selection, PCA reduction, and binary classification with rigorous nested cross-validation and permutation testing.

## Features

- **Data preprocessing**: log₂-transform, variance filter (top 100 genes)  
- **Feature selection**: ElasticNetCV-based filter  
- **Dimensionality reduction**: PCA to 2–5 components  
- **Modeling**: Nearest-centroid & logistic classifiers, NdLinear MLPs  
- **Evaluation**: Nested CV, early stopping, permutation tests, bootstrap CIs  

## Requirements

- Python 3.8+  
- [ndlinear](https://pypi.org/project/ndlinear)  
- pandas, numpy, scikit-learn, torch 

## Installation & Usage


### 1) Clone the repository
```bash
git clone https://github.com/tud03125/NdLinear-RNAseq-demo.git
cd NdLinear-RNAseq-demo
```

### 2) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # on Windows use: venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install ndlinear pandas numpy scikit-learn torch
```

### 4) Run the demo pipeline
```bash
python Applying_NdLinear_to_Bioinformatics.py
```

This will:

1. Load Salmon counts (salmon.merged.gene_counts.tsv) and sample sheet.

2. Log₂-transform + variance filter (top 100 genes).

3. ElasticNet feature selection on scaled data.

4. PCA reduction to 2–5 components.

5. Split into train/val/test (80/10/10%).

6. Nested CV to select hyperparameters.

7. Final training + test evaluation with bootstrap CIs and permutation test.

## Results

After running, you’ll see:

* Fold-by-fold CV accuracies

* Test accuracy + 95% bootstrap CI

* Permutation test p-value

## Caveats & Limitations

> ⚠️ **High-Dimension, Low-Sample-Size (HDLSS) Warning**  
> This demo uses only 50 samples but thousands of gene features. In such HDLSS settings, even extremely simple models (e.g. nearest-centroid on 2 PCA axes) can “memorize” the data and achieve 100 % accuracy without learning generalizable biology.  
>
> - **Overfitting risk:** Perfect cross-validation and test results here likely reflect dataset idiosyncrasies rather than a reproducible signal.  
> - **Permutation tests & bootstrap CIs:** We include permutation p-values and bootstrap confidence intervals to gauge how much of this performance may occur by chance.  
> - **Next steps:** To build robust models, consider transfer learning from large RNA-seq compendia, synthetic data augmentation, or collapsing features into known pathways.  

## Contributing

Contributions welcome! Please fork, open an issue, or submit a pull request.
Longer docs or tutorials can go in a docs/ directory or GitHub Wiki if this README grows too large

## License

This project is licensed under the MIT License (MIT).  

MIT License

Copyright (c) 2025 Michael Levin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell        
copies of the Software, and to permit persons to whom the Software is           
furnished to do so, subject to the following conditions:                        

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.                                 

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.                                                                       

## Acknowledgements

* Core linear layer logic from Ensemble-core/NdLinear (Apache-2.0) 

* This README structure follows GitHub’s “About READMEs” guidance 

* Best practices inspired by FreeCodeCamp’s README guide 
