# HydroNODE

NeuralODE models for hydrology

marvin.hoege@eawag.ch, Nov, 2022

Code repo for publication at https://hess.copernicus.org/articles/26/5085/2022/. When using the code, please cite.


### Installation
Install [Julia](https://julialang.org/downloads/), currently version 1.8.2 (for the 1st release of HydroNODE, v1.0.0, ideally use Julia version 1.7.2)

All required packages are installed automatically in a seperate environment (see https://pkgdocs.julialang.org/v1/toml-files/) when `HydroNODE_main.jl` is executed for the first time.

### Data
- download `CAMELS time series meteorology, observed flow, meta data (.zip)` from https://ral.ucar.edu/solutions/products/camels
- unzip and refer to folder `basin_dataset_public_v1p2` as `data_path` in `HydroNODE_main.jl`

### Train models
- Set user specific settings like `basin_id` in `HydroNODE_main.jl`
  and execute it.
