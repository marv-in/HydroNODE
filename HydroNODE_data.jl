# --------------------------------------------------
# HydroNODE Functions to work with the CAMELS data set
# - data loading and pre-processing
# - data formatting
#
# marvin.hoege@eawag.ch, Nov. 2022 (v1.1.0)
# --------------------------------------------------



function load_data(basin_id, data_path, source_data_set::String = "daymet")
    # function to load and preprocess camels data

    # make sure basin id is 8-digit string
    if !(typeof(basin_id) == String || length(basin_id) == 8)
        basin_id = lpad(basin_id, 8, "0")
    end

    # ==========================================================================
    # check if basin data is incomplete, flawed, otherwise problematic...

    # flagged ids from CAMELS due to missing hydrologic years
    # (see "dataset_summary.txt" in "CAMELS time series meteorology, observed flow, meta data (.zip)"
    #  at https://ral.ucar.edu/solutions/products/camels  )

    basins_with_missing_data = lpad.([01208990, 01613050, 02051000, 02235200, 02310947,
        02408540,02464146,03066000,03159540,03161000,03187500,03281100,03300400,03450000,
        05062500,05087500,06037500,06043500,06188000,06441500,07290650,07295000,07373000,
        07376000,07377000,08025500,08155200,09423350,09484000,09497800,09505200,10173450,
        10258000,12025000,12043000,12095000,12141300,12374250,13310700],8,"0")

    if in(basin_id, basins_with_missing_data)
        println("Missing data for selected basin. Adjustments required!")
    end


    # ==========================================================================
    # gauge_information has to be read first to obtain correct HUC (hydrologic unit code)
    path_gauge_meta = joinpath(data_path, "basin_metadata","gauge_information.txt")
    gauge_meta = CSV.File(path_gauge_meta, delim='\t', skipto=2, header = false)|> DataFrame

    all_basin_ids = lpad.(gauge_meta.Column2,8,"0")
    basin_huc = lpad(gauge_meta.Column1[all_basin_ids.==basin_id][1],2,"0")

    println('\n')
    println("Basin ID: "*basin_id, '\n')
    println("Hydrologic unit code (HUC): "*basin_huc, '\n')
    println("Forcings data set: "*source_data_set, '\n')

    path_forcing_data = joinpath(data_path, "basin_mean_forcing", source_data_set, basin_huc, basin_id*"_lump_cida_forcing_leap.txt")
    path_flow_data = joinpath(data_path, "usgs_streamflow", basin_huc, basin_id*"_streamflow_qc.txt")


    # =================================================================
    # Basin data
    # read first 3 lines with basin info:
    # 1. latitude of gauge
    # 2. elevation of gauge [m]
    # 3. basin area [m^2]
    basin_info_forcing = readdlm(path_forcing_data)[1:3]
    area = basin_info_forcing[3]

    println("Basin Data from forcings file:")
    println("Lat,    Elev,   Area [km^2]")
    println(round.(basin_info_forcing.*[1.0; 1.0; 10^-6], digits=2), '\n')


    gauge_meta_data = gauge_meta[findall(x -> x==parse(Int,basin_id),gauge_meta[:,2]),:]
    println("Gauging station name: ")
    println(gauge_meta_data[!,3][1], '\n')
    println(" Latitude,  Longitude    ")
    println(Matrix(gauge_meta_data[!,4:end-1]), '\n')


    # =================================================================
    # Flow data

    # read flow data, i.e. 5th column of .txt file
    data_flow = readdlm(path_flow_data)[:,5];
    data_flow_all = readdlm(path_flow_data)

    # filter out all false values
    data_flow_dates =  data_flow_all[findall(x -> x!=-999.0, data_flow),2:4]
    data_flow_dates = [Date(Dates.Year(x[1]), Dates.Month(x[2]), Dates.Day(x[3])) for x in eachrow(data_flow_dates)]

    # filter missing values
    filter!(x -> x != -999.0, data_flow)

    # transfer from cubic feet (304.8 mm) per second to mm/day (normalized by basin area)
    data_flow = data_flow.*(304.8^3)/(area*10^6) * 86400


    # =================================================================
    # Forcings data
    df = CSV.File(path_forcing_data, delim='\t', skipto=5, header = false) |> DataFrame

    df_dates = df[!,:Column1]
    df_date_length = length(df_dates[1])-3
    df_dates = Date.(SubString.(df_dates,1,df_date_length), "yyyy mm dd")
    df[!,:Column1] = df_dates

    # filter df values by available flow data dates
    filter!(row -> row.Column1 in data_flow_dates, df)

    # add processed columns to Dataframe
    insertcols!(df, ncol(df)+1, "Tmean(C)" => round.((df[!,:Column6].+df[!,:Column7])./2, digits =3));
    insertcols!(df, ncol(df)+1, Symbol("Flow(mm/s)") => data_flow);

    # rename columns
    col_names = ["Date", "Daylight(s)", "Prec(mm/day)","Srad(W/m2)","SWE(mm)","Tmax(C)","Tmin(C)","Vp(Pa)"]
    renaming = [names(df)[i] => col_names[i] for i in 1:8]
    rename!(df, renaming);

    df[!,Symbol("Daylight(s)")] = df[!,Symbol("Daylight(s)")]./3600
    rename!(df, "Daylight(s)" => "Daylight(h)");

    return df
end



function prepare_data(df, train_test_windows::NTuple{4,Date}, x_var=names(df)[2:end-1], y_var=names(df)[end])
    # format data

    train_start_date, train_stop_date, test_start_date, test_stop_date = train_test_windows

    train_data = filter(row -> row.Date in collect(train_start_date:Day(1):train_stop_date), df)
    test_data = filter(row -> row.Date in collect(test_start_date:Day(1):test_stop_date), df)
    all_data = filter(row -> row.Date in collect(train_start_date:Day(1):test_stop_date), df)

    train_x = Matrix(train_data[:,x_var])
    train_y = Vector(train_data[:,y_var])
    test_x = Matrix(test_data[:,x_var])
    test_y = Vector(test_data[:,y_var])
    data_x = Matrix(all_data[:,x_var])
    data_y = Vector(all_data[:,y_var])

    all_times = collect(train_start_date:Day(1):test_stop_date)

    train_timepoints = findall(x->x==train_start_date, all_times)[1]:findall(x->x==train_stop_date, all_times)[1]
    train_timepoints = collect((train_timepoints.-1.0).*1.0)

    test_timepoints = findall(x->x==test_start_date, all_times)[1]:findall(x->x==test_stop_date, all_times)[1]
    test_timepoints = collect((test_timepoints.-1.0).*1.0)

    data_timepoints = findall(x->x==train_start_date, all_times)[1]:findall(x->x==test_stop_date, all_times)[1]
    data_timepoints = collect((data_timepoints.-1.0).*1.0)

    println("Input variables: ")
    println(x_var, '\n')
    println("Output variable: ")
    println(y_var, '\n')
    println("Size: train_x=", size(train_x), " // train_y=", size(train_y))
    println("Size: test_x= ", size(test_x), " //  test_y= ", size(test_y), '\n')

    return data_x, data_y, data_timepoints, train_x, train_y, train_timepoints, test_x, test_y, test_timepoints

end
