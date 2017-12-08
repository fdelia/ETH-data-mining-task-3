# DataMining_Project
Repository for the Data Mining project, ETH Zurich, HerbstSemester 2017

# Usage
**Cross validation**
Use `runner_cv.py`. Usage is same as with `runner.py`, e.g. `python2.7 runner_cv.py data/handout_train.npy data/handout_test.npy example.py`.  
Has additional parameter `-K` which is the number of folds (default 3).

**Learning curve plot**
Use `runner_score.py`. Usage is same as with `runner.py`, e.g. `python2.7 runner_score.py data/handout_train.npy data/handout_test.npy example.py`.  
Additional parameter `-d` which is the "divider". E.g. with divider "3" samples are split into 0.33, 0.66 and 1.0 as factor of training samples. Higher divider gives a smoother learning curve, but runs longer. The algorithm is retrained on every step. Default is 5.
