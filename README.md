
This repository contains the raw data and analysis scripts for the results reported in

Travers, E., Rolisin, J. J. & Feeney, A. (2016). [The time course of conflict on the Cognitive Reflection Test](http://www.sciencedirect.com/science/article/pii/S0010027716300142). Cognition, 150, 109-118.

This consists of the following files

- `data/raw_data.csv`
  The raw data itself.
  This is processed by 
  - `preprocess.py`
    the python script that prepares this data for analysis, which depends on
    - `stimuli.py`
      which contains the coding data for each trial.
- The analyses themselves are contained in
    - `ByTrial.Rmd`
      looking at summary measures (resposne time, path length, etc.) for each trial
    - `TimeCourse.Rmd`
      fitting the full set of growth curve models to the time course data (this takes about 20 minutes on my machine)
    - `TimeCourse_short.Rmd`
      fitting only the important models (i.e. not testing differences between participants, or between problems)
    - `Functions.R`
      containing some useful common code used in the other scripts.
