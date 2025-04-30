<p align="center">
  <img src="ensemble_logo.jpg" alt="Logo" width="400">
  <br /> <br / >
</p>

<!--
  README.md for NdLinear-RNAseq-demo  
  Demonstrates application of NdLinear to human liver RNA-seq (GSE126848).
-->

## Summary

This repository demonstrates how to apply the [NdLinear](https://github.com/ensemble-core/NdLinear) library to bulk RNA-seq data (GSE126848), covering data loading, variance filtering, ElasticNet feature selection, PCA reduction, and binary classification with rigorous nested cross-validation and permutation testing. :contentReference[oaicite:3]{index=3}

## Features

- **Data preprocessing**: log₂-transform, variance filter (top 100 genes)  
- **Feature selection**: ElasticNetCV-based filter  
- **Dimensionality reduction**: PCA to 2–5 components  
- **Modeling**: Nearest-centroid & logistic classifiers, NdLinear MLPs  
- **Evaluation**: Nested CV, early stopping, permutation tests, bootstrap CIs  

## Requirements

- Python 3.8+  
- [ndlinear](https://pypi.org/project/ndlinear)  
- pandas, numpy, scikit-learn, torch :contentReference[oaicite:4]{index=4}  

## Installation & Usage


# 1) Clone the repository
```bash
git clone https://github.com/tud03125/NdLinear-RNAseq-demo.git
cd NdLinear-RNAseq-demo
```

# 2) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # on Windows use: venv\Scripts\activate
```

# 3) Install dependencies
```bash
pip install ndlinear pandas numpy scikit-learn torch
```

# 4) Run the demo pipeline
```bash
python Applying_NdLinear_to_Bioinformatics.py
```

This will:

Load Salmon counts (salmon.merged.gene_counts.tsv) and sample sheet.

Log₂-transform + variance filter (top 100 genes).

ElasticNet feature selection on scaled data.

PCA reduction to 2–5 components.

Split into train/val/test (80/10/10%).

Nested CV to select hyperparameters.

Final training + test evaluation with bootstrap CIs and permutation test.

## Results

After running, you’ll see:

Fold-by-fold CV accuracies

Test accuracy + 95% bootstrap CI

Permutation test p-value

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

-- Core linear layer logic from Ensemble-core/NdLinear (Apache-2.0) 

-- This README structure follows GitHub’s “About READMEs” guidance 

-- Best practices inspired by FreeCodeCamp’s README guide 
