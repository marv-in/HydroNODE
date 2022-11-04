# --------------------------------------------------
# HydroNODE Model Training
#
# pre-train NNs for single processes to make them learn the basic
# physical relations between input variables and process output
#
# marvin.hoege@eawag.ch, Nov. 2022 (v1.1.0)
# --------------------------------------------------

function prep_norm(norm_moments)
  val -> (val .- norm_moments[1])./norm_moments[2]
end


function pretrain_submodel(model, input, p_init, target_data; optmzr = ADAM(0.01), max_N_iter = 300)
  # pretrain NN via sum of least squares

  function prep_pred_NN_pretrain(model_, input_)
      (params) -> model_(input_,params)
  end

  pred_NN_pretrain_fct = prep_pred_NN_pretrain(model, permutedims(input))

  function loss_NN_pretrain(params, batch)
      sum((pred_NN_pretrain_fct(params)'.-batch).^2)
  end

  optf = Optimization.OptimizationFunction((θ, p) -> loss_NN_pretrain(θ, target_data), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, p_init)
  sol = Optimization.solve(optprob, optmzr, maxiters = max_N_iter)

  pred_opt = pred_NN_pretrain_fct(sol)'

  return sol, pred_opt

end


function pretrain_NNs_for_bucket_processes(chosen_model_id, NNs_in_fct, p_NNs_init, NN_input, p_bucket,S0_bucket, S1_bucket, Lday_bucket, P_bucket, T_bucket)

  f_bucket    = p_bucket[1]
  Smax_bucket = p_bucket[2]
  Qmax_bucket = p_bucket[3]
  Df_bucket   = p_bucket[4]
  Tmax_bucket = p_bucket[5]
  Tmin_bucket = p_bucket[6]

  # evapotranspiration
  ET_mech = ET.(S1_bucket, T_bucket, Lday_bucket, Smax_bucket)
  ET_mech = [x<0.0 ? 0.000000001 : x for x in ET_mech]

  # discharge
  Q_mech = Qb.(S1_bucket,f_bucket,Smax_bucket,Qmax_bucket) .+ Qs.(S1_bucket, Smax_bucket)
  Q_mech = [x<0.0 ? 0.000000001 : x for x in Q_mech]

  # snow precipitation
  Ps_mech = Ps.(P_bucket, T_bucket, Tmin_bucket)

  # rain precipitation
  Pr_mech = Pr.(P_bucket, T_bucket, Tmin_bucket)


  if chosen_model_id == "M50"
    NN_in_fct101 = NNs_in_fct[1]
    p_NN_init_101 = p_NNs_init[1]

    NN_in_fct102 = NNs_in_fct[2]
    p_NN_init_102 = p_NNs_init[2]

    # NN for ET
    p_NN_precal_101, = pretrain_submodel(NN_in_fct101, NN_input[:, [1,2,4]], p_NN_init_101,
      log.(ET_mech./Lday_bucket);
      optmzr = ADAM(0.01), max_N_iter = 300)

    # NN for Q
    p_NN_precal_102, = pretrain_submodel(NN_in_fct102, NN_input[:, [2,3]], p_NN_init_102,
      log.(Q_mech);
      optmzr = ADAM(0.01), max_N_iter = 300)

    p_NN_init_new = [p_NN_precal_101..., p_NN_precal_102...]


  elseif chosen_model_id == "M100"

    M_mech = minimum(hcat(S0_bucket, Df_bucket*(T_bucket.-Tmax_bucket)), dims = 2) .* step_fct.(T_bucket.-Tmax_bucket)
    M_mech = [x<0.0 ? 0.0 : x for x in M_mech]

    NN_in_fct101 = NNs_in_fct
    p_NN_init_101 = p_NNs_init

    p_NN_precal_101, = pretrain_submodel(NN_in_fct101, NN_input[:, [1,2,3,4]], p_NN_init_101,
    [log.(ET_mech./Lday_bucket) log.(Q_mech) asinh.(M_mech) asinh.(Ps_mech) asinh.(Pr_mech)];
    optmzr = ADAM(0.01), max_N_iter = 1000)

    p_NN_init_new = p_NN_precal_101

  end

  return p_NN_init_new

end


function train_model(pred_NODE, p_init, target_data, target_time; optmzr = ADAM(0.01), max_N_iter = 75)

  function prep_pred_model(time_batch)
      (p) -> pred_NODE(p, time_batch)[1]
  end

  pred_NN_model_fct = prep_pred_model(target_time)

  function loss_model(p)
      -NSE(pred_NN_model_fct(p),target_data)
  end

  callback = function (p,l)
      println("NSE: "*string(-l))
      return false
  end

  optf = Optimization.OptimizationFunction((θ, p) -> loss_model(θ), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, p_init)
  sol = Optimization.solve(optprob, optmzr, callback = callback, maxiters = max_N_iter)

  return sol

end
