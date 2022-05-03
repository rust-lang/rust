//Performs the entire benchmark process according to the documentation
template<class Input, class Output, class Parameters>
void run_benchmark(const char* module_path, const std::string& input_filepath, const std::string& output_prefix, const duration<double> minimum_measurable_time, const int nruns_F, const int nruns_J,
                   const duration<double> time_limit, const Parameters parameters) {

    const ModuleLoader module_loader(module_path);
    auto test = get_test<Input, Output>(module_loader);
    auto inputs = read_input_data<Input, Parameters>(input_filepath, parameters);

    test->prepare(std::move(inputs));

    const auto objective_time =
        measure_shortest_time(minimum_measurable_time, nruns_F, time_limit, *test, &ITest<Input, Output>::calculate_objective);

    const auto derivative_time =
        measure_shortest_time(minimum_measurable_time, nruns_J, time_limit, *test, &ITest<Input, Output>::calculate_jacobian);

    const auto output = test->output();

    const auto input_basename = filepath_to_basename(input_filepath);
    const auto module_basename = filepath_to_basename(module_path);

    save_time_to_file(output_prefix + input_basename + "_times_" + module_basename + ".txt", objective_time.count(), derivative_time.count());
    save_output_to_file(output, output_prefix, input_basename, module_basename);
}

template<class Input, class Output>
void run_benchmark(const char* const module_path, const std::string& input_filepath, const std::string& output_prefix, const duration<double> minimum_measurable_time, const int nruns_F, const int nruns_J,
                   const duration<double> time_limit) {
    run_benchmark<Input, Output, DefaultParameters>(module_path, input_filepath, output_prefix, minimum_measurable_time, nruns_F, nruns_J, time_limit, {});
}
