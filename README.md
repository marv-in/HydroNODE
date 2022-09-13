# HydroNODE

NeuralODE models for hydrology

marvin.hoege@eawag.ch, March, 2022

Code repo for publication at https://doi.org/10.5194/hess-2022-56. When using the code, please cite.



### Data

- download `CAMELS time series meteorology, observed flow, meta data (.zip)` from https://ral.ucar.edu/solutions/products/camels
- unzip and refer to folder `basin_dataset_public_v1p2` as `data_path` in `HydroNODE_main.jl`

### Installation
Install [Julia](https://julialang.org/downloads/) version 1.7 or newer. All required packages are installed automatically in a seperate environment when `HydroNODE_main.jl` is executed for the first time.

### Train models
- Set user specific settings like `basin_id` in `HydroNODE_main.jl`
  and execute it.
