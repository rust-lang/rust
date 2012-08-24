import option;

enum mode { mode_compile_fail, mode_run_fail, mode_run_pass, mode_pretty, }

type config = {
    // The library paths required for running the compiler
    compile_lib_path: ~str,

    // The library paths required for running compiled programs
    run_lib_path: ~str,

    // The rustc executable
    rustc_path: Path,

    // The directory containing the tests to run
    src_base: Path,

    // The directory where programs should be built
    build_base: Path,

    // Directory for auxiliary libraries
    aux_base: Path,

    // The name of the stage being built (stage1, etc)
    stage_id: ~str,

    // The test mode, compile-fail, run-fail, run-pass
    mode: mode,

    // Run ignored tests
    run_ignored: bool,

    // Only run tests that match this filter
    filter: option<~str>,

    // Write out a parseable log of tests that were run
    logfile: option<Path>,

    // A command line to prefix program execution with,
    // for running under valgrind
    runtool: option<~str>,

    // Flags to pass to the compiler
    rustcflags: option<~str>,

    // Explain what's going on
    verbose: bool};
