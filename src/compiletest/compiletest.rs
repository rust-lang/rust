// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "bin"]
#![feature(phase, slicing_syntax)]

#![deny(warnings)]

extern crate test;
extern crate getopts;
#[phase(plugin, link)] extern crate log;

extern crate regex;

use std::os;
use std::io;
use std::io::fs;
use std::from_str::FromStr;
use getopts::{optopt, optflag, reqopt};
use common::Config;
use common::{Pretty, DebugInfoGdb, DebugInfoLldb, Codegen};
use util::logv;
use regex::Regex;

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

pub fn parse_config(args: Vec<String> ) -> Config {

    let groups : Vec<getopts::OptGroup> =
        vec!(reqopt("", "compile-lib-path", "path to host shared libraries", "PATH"),
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
          optflag("", "ignored", "run tests marked as ignored"),
          optopt("", "runtool", "supervisor program to run tests under \
                                 (eg. emulator, valgrind)", "PROGRAM"),
          optopt("", "host-rustcflags", "flags to pass to rustc for host", "FLAGS"),
          optopt("", "target-rustcflags", "flags to pass to rustc for target", "FLAGS"),
          optflag("", "verbose", "run tests verbosely, showing all output"),
          optopt("", "logfile", "file to log test execution to", "FILE"),
          optopt("", "save-metrics", "file to save metrics to", "FILE"),
          optopt("", "ratchet-metrics", "file to ratchet metrics against", "FILE"),
          optopt("", "ratchet-noise-percent",
                 "percent change in metrics to consider noise", "N"),
          optflag("", "jit", "run tests under the JIT"),
          optopt("", "target", "the target to build for", "TARGET"),
          optopt("", "host", "the host to build for", "HOST"),
          optopt("", "gdb-version", "the version of GDB used", "MAJOR.MINOR"),
          optopt("", "android-cross-path", "Android NDK standalone path", "PATH"),
          optopt("", "adb-path", "path to the android debugger", "PATH"),
          optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH"),
          optopt("", "lldb-python-dir", "directory containing LLDB's python module", "PATH"),
          optopt("", "test-shard", "run shard A, of B shards, worth of the testsuite", "A.B"),
          optflag("h", "help", "show this message"));

    assert!(!args.is_empty());
    let argv0 = args[0].clone();
    let args_ = args.tail();
    if args[1].as_slice() == "-h" || args[1].as_slice() == "--help" {
        let message = format!("Usage: {} [OPTIONS] [TESTNAME...]", argv0);
        println!("{}", getopts::usage(message.as_slice(), groups.as_slice()));
        println!("");
        fail!()
    }

    let matches =
        &match getopts::getopts(args_.as_slice(), groups.as_slice()) {
          Ok(m) => m,
          Err(f) => fail!("{}", f)
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        let message = format!("Usage: {} [OPTIONS]  [TESTNAME...]", argv0);
        println!("{}", getopts::usage(message.as_slice(), groups.as_slice()));
        println!("");
        fail!()
    }

    fn opt_path(m: &getopts::Matches, nm: &str) -> Path {
        Path::new(m.opt_str(nm).unwrap())
    }

    let filter = if !matches.free.is_empty() {
        let s = matches.free[0].as_slice();
        match regex::Regex::new(s) {
            Ok(re) => Some(re),
            Err(e) => {
                println!("failed to parse filter /{}/: {}", s, e);
                fail!()
            }
        }
    } else {
        None
    };

    Config {
        compile_lib_path: matches.opt_str("compile-lib-path").unwrap(),
        run_lib_path: matches.opt_str("run-lib-path").unwrap(),
        rustc_path: opt_path(matches, "rustc-path"),
        clang_path: matches.opt_str("clang-path").map(|s| Path::new(s)),
        llvm_bin_path: matches.opt_str("llvm-bin-path").map(|s| Path::new(s)),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        aux_base: opt_path(matches, "aux-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode: FromStr::from_str(matches.opt_str("mode")
                                       .unwrap()
                                       .as_slice()).expect("invalid mode"),
        run_ignored: matches.opt_present("ignored"),
        filter: filter,
        cfail_regex: Regex::new(errors::EXPECTED_PATTERN).unwrap(),
        logfile: matches.opt_str("logfile").map(|s| Path::new(s)),
        save_metrics: matches.opt_str("save-metrics").map(|s| Path::new(s)),
        ratchet_metrics:
            matches.opt_str("ratchet-metrics").map(|s| Path::new(s)),
        ratchet_noise_percent:
            matches.opt_str("ratchet-noise-percent")
                   .and_then(|s| from_str::<f64>(s.as_slice())),
        runtool: matches.opt_str("runtool"),
        host_rustcflags: matches.opt_str("host-rustcflags"),
        target_rustcflags: matches.opt_str("target-rustcflags"),
        jit: matches.opt_present("jit"),
        target: opt_str2(matches.opt_str("target")),
        host: opt_str2(matches.opt_str("host")),
        gdb_version: extract_gdb_version(matches.opt_str("gdb-version")),
        android_cross_path: opt_path(matches, "android-cross-path"),
        adb_path: opt_str2(matches.opt_str("adb-path")),
        adb_test_dir: opt_str2(matches.opt_str("adb-test-dir")),
        adb_device_status:
            "arm-linux-androideabi" ==
                opt_str2(matches.opt_str("target")).as_slice() &&
            "(none)" !=
                opt_str2(matches.opt_str("adb-test-dir")).as_slice() &&
            !opt_str2(matches.opt_str("adb-test-dir")).is_empty(),
        lldb_python_dir: matches.opt_str("lldb-python-dir"),
        test_shard: test::opt_shard(matches.opt_str("test-shard")),
        verbose: matches.opt_present("verbose")
    }
}

pub fn log_config(config: &Config) {
    let c = config;
    logv(c, format!("configuration:"));
    logv(c, format!("compile_lib_path: {}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {}", config.run_lib_path));
    logv(c, format!("rustc_path: {}", config.rustc_path.display()));
    logv(c, format!("src_base: {}", config.src_base.display()));
    logv(c, format!("build_base: {}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", config.mode));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filter: {}",
                    opt_str(&config.filter
                                   .as_ref()
                                   .map(|re| {
                                       re.to_string().into_string()
                                   }))));
    logv(c, format!("runtool: {}", opt_str(&config.runtool)));
    logv(c, format!("host-rustcflags: {}",
                    opt_str(&config.host_rustcflags)));
    logv(c, format!("target-rustcflags: {}",
                    opt_str(&config.target_rustcflags)));
    logv(c, format!("jit: {}", config.jit));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("android-cross-path: {}",
                    config.android_cross_path.display()));
    logv(c, format!("adb_path: {}", config.adb_path));
    logv(c, format!("adb_test_dir: {}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}",
                    config.adb_device_status));
    match config.test_shard {
        None => logv(c, "test_shard: (all)".to_string()),
        Some((a,b)) => logv(c, format!("test_shard: {}.{}", a, b))
    }
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("\n"));
}

pub fn opt_str<'a>(maybestr: &'a Option<String>) -> &'a str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => s.as_slice(),
    }
}

pub fn opt_str2(maybestr: Option<String>) -> String {
    match maybestr {
        None => "(none)".to_string(),
        Some(s) => s,
    }
}

pub fn run_tests(config: &Config) {
    if config.target.as_slice() == "arm-linux-androideabi" {
        match config.mode {
            DebugInfoGdb => {
                println!("arm-linux-androideabi debug-info \
                         test uses tcp 5039 port. please reserve it");
            }
            _ =>{}
        }

        //arm-linux-androideabi debug-info test uses remote debugger
        //so, we test 1 task at once.
        // also trying to isolate problems with adb_run_wrapper.sh ilooping
        os::setenv("RUST_TEST_TASKS","1");
    }

    match config.mode {
        DebugInfoLldb => {
            // Some older versions of LLDB seem to have problems with multiple
            // instances running in parallel, so only run one test task at a
            // time.
            os::setenv("RUST_TEST_TASKS", "1");
        }
        _ => { /* proceed */ }
    }

    let opts = test_opts(config);
    let tests = make_tests(config);
    // sadly osx needs some file descriptor limits raised for running tests in
    // parallel (especially when we have lots and lots of child processes).
    // For context, see #8904
    io::test::raise_fd_limit();
    let res = test::run_tests_console(&opts, tests.into_iter().collect());
    match res {
        Ok(true) => {}
        Ok(false) => fail!("Some tests failed"),
        Err(e) => {
            println!("I/O failure during tests: {}", e);
        }
    }
}

pub fn test_opts(config: &Config) -> test::TestOpts {
    test::TestOpts {
        filter: match config.filter {
            None => None,
            Some(ref filter) => Some(filter.clone()),
        },
        run_ignored: config.run_ignored,
        logfile: config.logfile.clone(),
        run_tests: true,
        run_benchmarks: true,
        ratchet_metrics: config.ratchet_metrics.clone(),
        ratchet_noise_percent: config.ratchet_noise_percent.clone(),
        save_metrics: config.save_metrics.clone(),
        test_shard: config.test_shard.clone(),
        nocapture: false,
        color: test::AutoColor,
    }
}

pub fn make_tests(config: &Config) -> Vec<test::TestDescAndFn> {
    debug!("making tests from {}",
           config.src_base.display());
    let mut tests = Vec::new();
    let dirs = fs::readdir(&config.src_base).unwrap();
    for file in dirs.iter() {
        let file = file.clone();
        debug!("inspecting file {}", file.display());
        if is_test(config, &file) {
            let t = make_test(config, &file, || {
                match config.mode {
                    Codegen => make_metrics_test_closure(config, &file),
                    _ => make_test_closure(config, &file)
                }
            });
            tests.push(t)
        }
    }
    tests
}

pub fn is_test(config: &Config, testfile: &Path) -> bool {
    // Pretty-printer does not work with .rc files yet
    let valid_extensions =
        match config.mode {
          Pretty => vec!(".rs".to_string()),
          _ => vec!(".rc".to_string(), ".rs".to_string())
        };
    let invalid_prefixes = vec!(".".to_string(), "#".to_string(), "~".to_string());
    let name = testfile.filename_str().unwrap();

    let mut valid = false;

    for ext in valid_extensions.iter() {
        if name.ends_with(ext.as_slice()) {
            valid = true;
        }
    }

    for pre in invalid_prefixes.iter() {
        if name.starts_with(pre.as_slice()) {
            valid = false;
        }
    }

    return valid;
}

pub fn make_test(config: &Config, testfile: &Path, f: || -> test::TestFn)
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

pub fn make_test_name(config: &Config, testfile: &Path) -> test::TestName {

    // Try to elide redundant long paths
    fn shorten(path: &Path) -> String {
        let filename = path.filename_str();
        let p = path.dir_path();
        let dir = p.filename_str();
        format!("{}/{}", dir.unwrap_or(""), filename.unwrap_or(""))
    }

    test::DynTestName(format!("[{}] {}", config.mode, shorten(testfile)))
}

pub fn make_test_closure(config: &Config, testfile: &Path) -> test::TestFn {
    let config = (*config).clone();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let testfile = testfile.as_str().unwrap().to_string();
    test::DynTestFn(proc() {
        runtest::run(config, testfile)
    })
}

pub fn make_metrics_test_closure(config: &Config, testfile: &Path) -> test::TestFn {
    let config = (*config).clone();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let testfile = testfile.as_str().unwrap().to_string();
    test::DynMetricFn(proc(mm) {
        runtest::run_metrics(config, testfile, mm)
    })
}

fn extract_gdb_version(full_version_line: Option<String>) -> Option<String> {
    match full_version_line {
        Some(ref full_version_line)
          if full_version_line.as_slice().trim().len() > 0 => {
            let full_version_line = full_version_line.as_slice().trim();

            let re = Regex::new(r"(^|[^0-9])([0-9]\.[0-9])([^0-9]|$)").unwrap();

            match re.captures(full_version_line) {
                Some(captures) => {
                    Some(captures.at(2).to_string())
                }
                None => {
                    println!("Could not extract GDB version from line '{}'",
                             full_version_line);
                    None
                }
            }
        },
        _ => None
    }
}
