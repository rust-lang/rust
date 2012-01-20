import option;

enum mode { mode_compile_fail; mode_run_fail; mode_run_pass; mode_pretty; }

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
    {compile_lib_path: str,
     run_lib_path: str,
     rustc_path: str,
     src_base: str,
     build_base: str,
     stage_id: str,
     mode: mode,
     run_ignored: bool,
     filter: option::t<str>,
     runtool: option::t<str>,
     rustcflags: option::t<str>,
     verbose: bool};

type cx = {config: config, procsrv: procsrv::handle};
