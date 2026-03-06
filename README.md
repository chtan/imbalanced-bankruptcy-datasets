ipynb files
===

The ipynb files contain the code for training and testing the various pipelines discussed in the paper.

The correspondence between the pipelines and the files is as follows:

1. FS=Embedded_FS.ipynb : embedded feature selectors, no resampler

2. FS=Embedded_FS-Resampler.ipynb : embedded feature selectors, feature selector precedes resampler

3. FS=Embedded_Resampler-FS.ipynb : embedded feature selectors, feature selector follows resampler

4. FS=Filter_FS:ipynb : wrapper feature selector, no resampler

5. FS=Filter_FS-Resampler+Resampler-FS.ipynb : wrapper feature selector, resampler before or after feature selector

6. FS=Wrapper_FS.ipynb : wrapper feature selectors, no resampler

7. FS=Wrapper_FS-Resampler.ipynb : wrapper feature selectors, feature selector precedes resampler

8. FS=Wrapper_Resampler-FS.ipynb : wrapper feature selectors, feature selector follows resampler


Running these files produce the performance metrics statistics which are saved for the next step.


py files
===

After the ML pipeline testing is completed, the performance metrics statistics are processed to generate graphs and tables that are presented in the paper.

The Python script and the graph or table that it helps produce is as follows:

victories_by_resampler.py : Fig. 4

victories_by_fs.py : Fig. 6

victories_by_clf.py : Fig. 7

averageRanks.py : Fig. 5

overallAverageRanks.py : Table 6, Fig. 8

bayesianTesting.py : Fig. 13, Fig. 14