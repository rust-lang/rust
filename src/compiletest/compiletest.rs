// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_type = "bin"];

#[allow(non_camel_case_types)];
#[allow(unrecognized_lint)]; // NOTE: remove after snapshot
#[deny(warnings)];

extern mod extra;

use std::os;
use std::f64;

use extra::getopts;
use extra::getopts::groups::{optopt, optflag, reqopt};
use extra::test;

use common::config;
use common::mode_run_pass;
use common::mode_run_fail;
use common::mode_compile_fail;
use common::mode_pretty;
use common::mode_debug_info;
use common::mode_codegen;
use common::mode;
use util::logv;

pub mod procsrv;
pub mod util;
pub mod header;
pub mod runtest;
pub mod common;
pub mod errors;

pub fn main() {
    let args = os::args();
    let config = parse_config(args);
    log_config(&config);
    run_tests(&config);
}

pub fn parse_config(args: ~[~str]) -> config {

    let groups : ~[getopts::groups::OptGroup] =
        ~[reqopt("", "compile-lib-path", "path to host shared libraries", "PATH"),
          reqopt("", "run-lib-path", "path to target shared libraries", "PATH"),
          reqopt("", "rustc-path", "path to rustc to use for compiling", "PATH"),
          optopt("", "clang-path", "path to  executable for codegen tests", "PATH"),
          optopt("", "llvm-bin-path", "path to directory holding llvm binaries", "DIR"),
          reqopt("", "src-base", "directory to scan for test files", "PATH"),
          reqopt("", "build-base", "directory to deposit test outputs", "PATH"),
          reqopt("", "aux-base", "directory to find auxiliary test files", "PATH"),
          reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET"),
          reqopt("", "mode", "which sort of compile tests to run",
                 "(compile-fail|run-fail|run-pass|pretty|debug-info)"),
          optflag("", "ignored", "run tests marked as ignored / xfailed"),
          optopt("", "runtool", "supervisor program to run tests under \
                                 (eg. emulator, valgrind)", "PROGRAM"),
          optopt("", "rustcflags", "flags to pass to rustc", "FLAGS"),
          optflag("", "verbose", "run tests verbosely, showing all output"),
          optopt("", "logfile", "file to log test execution to", "FILE"),
          optopt("", "save-metrics", "file to save metrics to", "FILE"),
          optopt("", "ratchet-metrics", "file to ratchet metrics against", "FILE"),
          optopt("", "ratchet-noise-percent",
                 "percent change in metrics to consider noise", "N"),
          optflag("", "jit", "run tests under the JIT"),
          optflag("", "newrt", "run tests on the new runtime / scheduler"),
          optopt("", "target", "the target to build for", "TARGET"),
          optopt("", "adb-path", "path to the android debugger", "PATH"),
          optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH"),
          optflag("h", "help", "show this message"),
         ];

    assert!(!args.is_empty());
    let argv0 = args[0].clone();
    let args_ = args.tail();
    if args[1] == ~"-h" || args[1] == ~"--help" {
        let message = fmt!("Usage: %s [OPTIONS] [TESTNAME...]", argv0);
        println(getopts::groups::usage(message, groups));
        fail!()
    }

    let matches =
        &match getopts::groups::getopts(args_, groups) {
          Ok(m) => m,
          Err(f) => fail!(getopts::fail_str(f))
        };

    if getopts::opt_present(matches, "h") || getopts::opt_present(matches, "help") {
        let message = fmt!("Usage: %s [OPTIONS]  [TESTNAME...]", argv0);
        println(getopts::groups::usage(message, groups));
        fail!()
    }

    fn opt_path(m: &getopts::Matches, nm: &str) -> Path {
        Path(getopts::opt_str(m, nm))
    }

    config {
        compile_lib_path: getopts::opt_str(matches, "compile-lib-path"),
        run_lib_path: getopts::opt_str(matches, "run-lib-path"),
        rustc_path: opt_path(matches, "rustc-path"),
        clang_path: getopts::opt_maybe_str(matches, "clang-path").map(|s| Path(*s)),
        llvm_bin_path: getopts::opt_maybe_str(matches, "llvm-bin-path").map(|s| Path(*s)),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        aux_base: opt_path(matches, "aux-base"),
        stage_id: getopts::opt_str(matches, "stage-id"),
        mode: str_mode(getopts::opt_str(matches, "mode")),
        run_ignored: getopts::opt_present(matches, "ignored"),
        filter:
            if !matches.free.is_empty() {
                 Some(matches.free[0].clone())
            } else {
                None
            },
        logfile: getopts::opt_maybe_str(matches, "logfile").map(|s| Path(*s)),
        save_metrics: getopts::opt_maybe_str(matches, "save-metrics").map(|s| Path(*s)),
        ratchet_metrics:
            getopts::opt_maybe_str(matches, "ratchet-metrics").map(|s| Path(*s)),
        ratchet_noise_percent:
            getopts::opt_maybe_str(matches,
                                   "ratchet-noise-percent").map(|s|
                                                                f64::from_str(*s).get()),
        runtool: getopts::opt_maybe_str(matches, "runtool"),
        rustcflags: getopts::opt_maybe_str(matches, "rustcflags"),
        jit: getopts::opt_present(matches, "jit"),
        newrt: getopts::opt_present(matches, "newrt"),
        target: opt_str2(getopts::opt_maybe_str(matches, "target")).to_str(),
        adb_path: opt_str2(getopts::opt_maybe_str(matches, "adb-path")).to_str(),
        adb_test_dir:
            opt_str2(getopts::opt_maybe_str(matches, "adb-test-dir")).to_str(),
        adb_device_status:
            if (opt_str2(getopts::opt_maybe_str(matches, "target")) ==
                ~"arm-linux-androideabi") {
                if (opt_str2(getopts::opt_maybe_str(matches, "adb-test-dir")) !=
                    ~"(none)" &&
                    opt_str2(getopts::opt_maybe_str(matches, "adb-test-dir")) !=
                    ~"") { true }
                else { false }
            } else { false },
        verbose: getopts::opt_present(matches, "verbose")
    }
}

pub fn log_config(config: &config) {
    let c = config;
    logv(c, fmt!("configuration:"));
    logv(c, fmt!("compile_lib_path: %s", config.compile_lib_path));
    logv(c, fmt!("run_lib_path: %s", config.run_lib_path));
    logv(c, fmt!("rustc_path: %s", config.rustc_path.to_str()));
    logv(c, fmt!("src_base: %s", config.src_base.to_str()));
    logv(c, fmt!("build_base: %s", config.build_base.to_str()));
    logv(c, fmt!("stage_id: %s", config.stage_id));
    logv(c, fmt!("mode: %s", mode_str(config.mode)));
    logv(c, fmt!("run_ignored: %b", config.run_ignored));
    logv(c, fmt!("filter: %s", opt_str(&config.filter)));
    logv(c, fmt!("runtool: %s", opt_str(&config.runtool)));
    logv(c, fmt!("rustcflags: %s", opt_str(&config.rustcflags)));
    logv(c, fmt!("jit: %b", config.jit));
    logv(c, fmt!("newrt: %b", config.newrt));
    logv(c, fmt!("target: %s", config.target));
    logv(c, fmt!("adb_path: %s", config.adb_path));
    logv(c, fmt!("adb_test_dir: %s", config.adb_test_dir));
    logv(c, fmt!("adb_device_status: %b", config.adb_device_status));
    logv(c, fmt!("verbose: %b", config.verbose));
    logv(c, fmt!("\n"));
}

pub fn opt_str<'a>(maybestr: &'a Option<~str>) -> &'a str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => {
            let s: &'a str = *s;
            s
        }
    }
}

pub fn opt_str2(maybestr: Option<~str>) -> ~str {
    match maybestr { None => ~"(none)", Some(s) => { s } }
}

pub fn str_opt(maybestr: ~str) -> Option<~str> {
    if maybestr != ~"(none)" { Some(maybestr) } else { None }
}

pub fn str_mode(s: ~str) -> mode {
    match s {
      ~"compile-fail" => mode_compile_fail,
      ~"run-fail" => mode_run_fail,
      ~"run-pass" => mode_run_pass,
      ~"pretty" => mode_pretty,
      ~"debug-info" => mode_debug_info,
      ~"codegen" => mode_codegen,
      _ => fail!("invalid mode")
    }
}

pub fn mode_str(mode: mode) -> ~str {
    match mode {
      mode_compile_fail => ~"compile-fail",
      mode_run_fail => ~"run-fail",
      mode_run_pass => ~"run-pass",
      mode_pretty => ~"pretty",
      mode_debug_info => ~"debug-info",
      mode_codegen => ~"codegen",
    }
}

pub fn run_tests(config: &config) {
    let opts = test_opts(config);
    let tests = make_tests(config);
    let res = test::run_tests_console(&opts, tests);
    if !res { fail!("Some tests failed"); }
}

pub fn test_opts(config: &config) -> test::TestOpts {
    test::TestOpts {
        filter: config.filter.clone(),
        run_ignored: config.run_ignored,
        logfile: config.logfile.clone(),
        run_tests: true,
        run_benchmarks: true,
        ratchet_metrics: copy config.ratchet_metrics,
        ratchet_noise_percent: copy config.ratchet_noise_percent,
        save_metrics: copy config.save_metrics,
    }
}

pub fn make_tests(config: &config) -> ~[test::TestDescAndFn] {
    debug!("making tests from %s",
           config.src_base.to_str());
    let mut tests = ~[];
    let dirs = os::list_dir_path(&config.src_base);
    for dirs.iter().advance |file| {
        let file = (*file).clone();
        debug!("inspecting file %s", file.to_str());
        if is_test(config, file) {
            let t = do make_test(config, file) {
                match config.mode {
                    mode_codegen => make_metrics_test_closure(config, file),
                    _ => make_test_closure(config, file)
                }
            };
            tests.push(t)
        }
    }
    tests
}

pub fn is_test(config: &config, testfile: &Path) -> bool {
    // Pretty-printer does not work with .rc files yet
    let valid_extensions =
        match config.mode {
          mode_pretty => ~[~".rs"],
          _ => ~[~".rc", ~".rs"]
        };
    let invalid_prefixes = ~[~".", ~"#", ~"~"];
    let name = testfile.filename().get();

    let mut valid = false;

    for valid_extensions.iter().advance |ext| {
        if name.ends_with(*ext) { valid = true; }
    }

    for invalid_prefixes.iter().advance |pre| {
        if name.starts_with(*pre) { valid = false; }
    }

    return valid;
}

pub fn make_test(config: &config, testfile: &Path,
                 f: &fn()->test::TestFn) -> test::TestDescAndFn {
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: make_test_name(config, testfile),
            ignore: header::is_test_ignored(config, testfile),
            should_fail: false
        },
        testfn: f(),
    }
}

pub fn make_test_name(config: &config, testfile: &Path) -> test::TestName {

    // Try to elide redundant long paths
    fn shorten(path: &Path) -> ~str {
        let filename = path.filename();
        let dir = path.pop().filename();
        fmt!("%s/%s", dir.get_or_default(~""), filename.get_or_default(~""))
    }

    test::DynTestName(fmt!("[%s] %s",
                           mode_str(config.mode),
                           shorten(testfile)))
}

pub fn make_test_closure(config: &config, testfile: &Path) -> test::TestFn {
    use std::cell::Cell;
    let config = Cell::new((*config).clone());
    let testfile = Cell::new(testfile.to_str());
    test::DynTestFn(|| { runtest::run(config.take(), testfile.take()) })
}

pub fn make_metrics_test_closure(config: &config, testfile: &Path) -> test::TestFn {
    use std::cell::Cell;
    let config = Cell::new(copy *config);
    let testfile = Cell::new(testfile.to_str());
    test::DynMetricFn(|mm| { runtest::run_metrics(config.take(), testfile.take(), mm) })
}
