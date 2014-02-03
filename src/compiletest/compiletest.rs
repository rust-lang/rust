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
#[deny(warnings)];

extern mod extra;

use std::os;
use std::io;
use std::io::fs;

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
          optopt("", "target", "the target to build for", "TARGET"),
          optopt("", "host", "the host to build for", "HOST"),
          optopt("", "adb-path", "path to the android debugger", "PATH"),
          optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH"),
          optopt("", "test-shard", "run shard A, of B shards, worth of the testsuite", "A.B"),
          optflag("h", "help", "show this message"),
         ];

    assert!(!args.is_empty());
    let argv0 = args[0].clone();
    let args_ = args.tail();
    if args[1] == ~"-h" || args[1] == ~"--help" {
        let message = format!("Usage: {} [OPTIONS] [TESTNAME...]", argv0);
        println!("{}", getopts::groups::usage(message, groups));
        println!("");
        fail!()
    }

    let matches =
        &match getopts::groups::getopts(args_, groups) {
          Ok(m) => m,
          Err(f) => fail!("{}", f.to_err_msg())
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        let message = format!("Usage: {} [OPTIONS]  [TESTNAME...]", argv0);
        println!("{}", getopts::groups::usage(message, groups));
        println!("");
        fail!()
    }

    fn opt_path(m: &getopts::Matches, nm: &str) -> Path {
        Path::new(m.opt_str(nm).unwrap())
    }

    config {
        compile_lib_path: matches.opt_str("compile-lib-path").unwrap(),
        run_lib_path: matches.opt_str("run-lib-path").unwrap(),
        rustc_path: opt_path(matches, "rustc-path"),
        clang_path: matches.opt_str("clang-path").map(|s| Path::new(s)),
        llvm_bin_path: matches.opt_str("llvm-bin-path").map(|s| Path::new(s)),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        aux_base: opt_path(matches, "aux-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode: str_mode(matches.opt_str("mode").unwrap()),
        run_ignored: matches.opt_present("ignored"),
        filter:
            if !matches.free.is_empty() {
                 Some(matches.free[0].clone())
            } else {
                None
            },
        logfile: matches.opt_str("logfile").map(|s| Path::new(s)),
        save_metrics: matches.opt_str("save-metrics").map(|s| Path::new(s)),
        ratchet_metrics:
            matches.opt_str("ratchet-metrics").map(|s| Path::new(s)),
        ratchet_noise_percent:
            matches.opt_str("ratchet-noise-percent").and_then(|s| from_str::<f64>(s)),
        runtool: matches.opt_str("runtool"),
        rustcflags: matches.opt_str("rustcflags"),
        jit: matches.opt_present("jit"),
        target: opt_str2(matches.opt_str("target")).to_str(),
        host: opt_str2(matches.opt_str("host")).to_str(),
        adb_path: opt_str2(matches.opt_str("adb-path")).to_str(),
        adb_test_dir:
            opt_str2(matches.opt_str("adb-test-dir")).to_str(),
        adb_device_status:
            "arm-linux-androideabi" == opt_str2(matches.opt_str("target")) &&
            "(none)" != opt_str2(matches.opt_str("adb-test-dir")) &&
            !opt_str2(matches.opt_str("adb-test-dir")).is_empty(),
        test_shard: test::opt_shard(matches.opt_str("test-shard")),
        verbose: matches.opt_present("verbose")
    }
}

pub fn log_config(config: &config) {
    let c = config;
    logv(c, format!("configuration:"));
    logv(c, format!("compile_lib_path: {}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {}", config.run_lib_path));
    logv(c, format!("rustc_path: {}", config.rustc_path.display()));
    logv(c, format!("src_base: {}", config.src_base.display()));
    logv(c, format!("build_base: {}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", mode_str(config.mode)));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filter: {}", opt_str(&config.filter)));
    logv(c, format!("runtool: {}", opt_str(&config.runtool)));
    logv(c, format!("rustcflags: {}", opt_str(&config.rustcflags)));
    logv(c, format!("jit: {}", config.jit));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("adb_path: {}", config.adb_path));
    logv(c, format!("adb_test_dir: {}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}", config.adb_device_status));
    match config.test_shard {
        None => logv(c, ~"test_shard: (all)"),
        Some((a,b)) => logv(c, format!("test_shard: {}.{}", a, b))
    }
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("\n"));
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
    if config.target == ~"arm-linux-androideabi" {
        match config.mode{
            mode_debug_info => {
                println!("arm-linux-androideabi debug-info \
                         test uses tcp 5039 port. please reserve it");
                //arm-linux-androideabi debug-info test uses remote debugger
                //so, we test 1 task at once
                os::setenv("RUST_TEST_TASKS","1");
            }
            _ =>{}
        }
    }

    let opts = test_opts(config);
    let tests = make_tests(config);
    // sadly osx needs some file descriptor limits raised for running tests in
    // parallel (especially when we have lots and lots of child processes).
    // For context, see #8904
    io::test::raise_fd_limit();
    let res = test::run_tests_console(&opts, tests);
    match res {
        Ok(true) => {}
        Ok(false) => fail!("Some tests failed"),
        Err(e) => {
            println!("I/O failure during tests: {}", e);
        }
    }
}

pub fn test_opts(config: &config) -> test::TestOpts {
    test::TestOpts {
        filter: config.filter.clone(),
        run_ignored: config.run_ignored,
        logfile: config.logfile.clone(),
        run_tests: true,
        run_benchmarks: true,
        ratchet_metrics: config.ratchet_metrics.clone(),
        ratchet_noise_percent: config.ratchet_noise_percent.clone(),
        save_metrics: config.save_metrics.clone(),
        test_shard: config.test_shard.clone()
    }
}

pub fn make_tests(config: &config) -> ~[test::TestDescAndFn] {
    debug!("making tests from {}",
           config.src_base.display());
    let mut tests = ~[];
    let dirs = fs::readdir(&config.src_base).unwrap();
    for file in dirs.iter() {
        let file = file.clone();
        debug!("inspecting file {}", file.display());
        if is_test(config, &file) {
            let t = make_test(config, &file, || {
                match config.mode {
                    mode_codegen => make_metrics_test_closure(config, &file),
                    _ => make_test_closure(config, &file)
                }
            });
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
    let name = testfile.filename_str().unwrap();

    let mut valid = false;

    for ext in valid_extensions.iter() {
        if name.ends_with(*ext) { valid = true; }
    }

    for pre in invalid_prefixes.iter() {
        if name.starts_with(*pre) { valid = false; }
    }

    return valid;
}

pub fn make_test(config: &config, testfile: &Path, f: || -> test::TestFn)
                 -> test::TestDescAndFn {
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
        let filename = path.filename_str();
        let p = path.dir_path();
        let dir = p.filename_str();
        format!("{}/{}", dir.unwrap_or(""), filename.unwrap_or(""))
    }

    test::DynTestName(format!("[{}] {}",
                              mode_str(config.mode),
                              shorten(testfile)))
}

pub fn make_test_closure(config: &config, testfile: &Path) -> test::TestFn {
    let config = (*config).clone();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let testfile = testfile.as_str().unwrap().to_owned();
    test::DynTestFn(proc() { runtest::run(config, testfile) })
}

pub fn make_metrics_test_closure(config: &config, testfile: &Path) -> test::TestFn {
    let config = (*config).clone();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let testfile = testfile.as_str().unwrap().to_owned();
    test::DynMetricFn(proc(mm) {
        runtest::run_metrics(config, testfile, mm)
    })
}
