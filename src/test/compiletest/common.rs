import std::option;

tag mode { mode_compile_fail; mode_run_fail; mode_run_pass; mode_pretty; }

type config =
    // The library paths required for running the compiler
    // The library paths required for running compiled programs
    // The rustc executable
    // The directory containing the tests to run
    // The directory where programs should be built
    // The name of the stage being built (stage1, etc)
    // The test mode, compile-fail, run-fail, run-pass
    // Run ignored tests
    // Only run tests that match this filter
    // A command line to prefix program execution with,
    // for running under valgrind
    // Flags to pass to the compiler
    // Explain what's going on
    {compile_lib_path: istr,
     run_lib_path: istr,
     rustc_path: istr,
     src_base: istr,
     build_base: istr,
     stage_id: istr,
     mode: mode,
     run_ignored: bool,
     filter: option::t<istr>,
     runtool: option::t<istr>,
     rustcflags: option::t<istr>,
     verbose: bool};

type cx = {config: config, procsrv: procsrv::handle};
