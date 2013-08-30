// Copyright 2012-2013 The Rust Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone, Eq)]
pub enum mode {
    mode_compile_fail,
    mode_run_fail,
    mode_run_pass,
    mode_pretty,
    mode_debug_info,
    mode_codegen
}

#[deriving(Clone)]
pub struct config {
    // The library paths required for running the compiler
    compile_lib_path: ~str,

    // The library paths required for running compiled programs
    run_lib_path: ~str,

    // The rustc executable
    rustc_path: Path,

    // The clang executable
    clang_path: Option<Path>,

    // The llvm binaries path
    llvm_bin_path: Option<Path>,

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
    filter: Option<~str>,

    // Write out a parseable log of tests that were run
    logfile: Option<Path>,

    // Write out a json file containing any metrics of the run
    save_metrics: Option<Path>,

    // Write and ratchet a metrics file
    ratchet_metrics: Option<Path>,

    // Percent change in metrics to consider noise
    ratchet_noise_percent: Option<f64>,

    // "Shard" of the testsuite to run: this has the form of
    // two numbers (a,b), and causes only those tests with
    // positional order equal to a mod b to run.
    test_shard: Option<(uint,uint)>,

    // A command line to prefix program execution with,
    // for running under valgrind
    runtool: Option<~str>,

    // Flags to pass to the compiler
    rustcflags: Option<~str>,

    // Run tests using the JIT
    jit: bool,

    // Target system to be tested
    target: ~str,

    // Extra parameter to run adb on arm-linux-androideabi
    adb_path: ~str,

    // Extra parameter to run test sute on arm-linux-androideabi
    adb_test_dir: ~str,

    // status whether android device available or not
    adb_device_status: bool,

    // Explain what's going on
    verbose: bool

}
