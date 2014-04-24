// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
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
    mode_debug_info_gdb,
    mode_debug_info_lldb,
    mode_codegen
}

#[deriving(Clone)]
pub struct config {
    // The library paths required for running the compiler
    pub compile_lib_path: ~str,

    // The library paths required for running compiled programs
    pub run_lib_path: ~str,

    // The rustc executable
    pub rustc_path: Path,

    // The clang executable
    pub clang_path: Option<Path>,

    // The llvm binaries path
    pub llvm_bin_path: Option<Path>,

    // The directory containing the tests to run
    pub src_base: Path,

    // The directory where programs should be built
    pub build_base: Path,

    // Directory for auxiliary libraries
    pub aux_base: Path,

    // The name of the stage being built (stage1, etc)
    pub stage_id: ~str,

    // The test mode, compile-fail, run-fail, run-pass
    pub mode: mode,

    // Run ignored tests
    pub run_ignored: bool,

    // Only run tests that match this filter
    pub filter: Option<~str>,

    // Write out a parseable log of tests that were run
    pub logfile: Option<Path>,

    // Write out a json file containing any metrics of the run
    pub save_metrics: Option<Path>,

    // Write and ratchet a metrics file
    pub ratchet_metrics: Option<Path>,

    // Percent change in metrics to consider noise
    pub ratchet_noise_percent: Option<f64>,

    // "Shard" of the testsuite to pub run: this has the form of
    // two numbers (a,b), and causes only those tests with
    // positional order equal to a mod b to run.
    pub test_shard: Option<(uint,uint)>,

    // A command line to prefix program execution with,
    // for running under valgrind
    pub runtool: Option<~str>,

    // Flags to pass to the compiler when building for the host
    pub host_rustcflags: Option<~str>,

    // Flags to pass to the compiler when building for the target
    pub target_rustcflags: Option<~str>,

    // Run tests using the JIT
    pub jit: bool,

    // Target system to be tested
    pub target: ~str,

    // Host triple for the compiler being invoked
    pub host: ~str,

    // Extra parameter to run adb on arm-linux-androideabi
    pub adb_path: ~str,

    // Extra parameter to run test sute on arm-linux-androideabi
    pub adb_test_dir: ~str,

    // status whether android device available or not
    pub adb_device_status: bool,

    // the path containing LLDB's Python module
    pub lldb_python_dir: Option<~str>,

    // Explain what's going on
    pub verbose: bool

}
