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

#![feature(box_syntax)]
#![feature(collections)]
#![feature(int_uint)]
#![feature(old_io)]
#![feature(old_path)]
#![feature(rustc_private)]
#![feature(unboxed_closures)]
#![feature(std_misc)]
#![feature(test)]
#![feature(unicode)]
#![feature(env)]
#![feature(core)]

#![deny(warnings)]

extern crate test;
extern crate getopts;

#[macro_use]
extern crate log;

use std::env;
use std::old_io;
use std::old_io::fs;
use std::thunk::Thunk;
use getopts::{optopt, optflag, reqopt};
use common::Config;
use common::{Pretty, DebugInfoGdb, DebugInfoLldb, Codegen};
use util::logv;

pub mod procsrv;
pub mod util;
pub mod header;
pub mod runtest;
pub mod common;
pub mod errors;

pub fn main() {
    let config = parse_config(env::args().collect());

    if config.valgrind_path.is_none() && config.force_valgrind {
        panic!("Can't find Valgrind to run Valgrind tests");
    }

    log_config(&config);
    run_tests(&config);
}

pub fn parse_config(args: Vec<String> ) -> Config {

    let groups : Vec<getopts::OptGroup> =
        vec!(reqopt("", "compile-lib-path", "path to host shared libraries", "PATH"),
          reqopt("", "run-lib-path", "path to target shared libraries", "PATH"),
          reqopt("", "rustc-path", "path to rustc to use for compiling", "PATH"),
          optopt("", "clang-path", "path to  executable for codegen tests", "PATH"),
          optopt("", "valgrind-path", "path to Valgrind executable for Valgrind tests", "PROGRAM"),
          optflag("", "force-valgrind", "fail if Valgrind tests cannot be run under Valgrind"),
          optopt("", "llvm-bin-path", "path to directory holding llvm binaries", "DIR"),
          reqopt("", "src-base", "directory to scan for test files", "PATH"),
          reqopt("", "build-base", "directory to deposit test outputs", "PATH"),
          reqopt("", "aux-base", "directory to find auxiliary test files", "PATH"),
          reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET"),
          reqopt("", "mode", "which sort of compile tests to run",
                 "(compile-fail|parse-fail|run-fail|run-pass|run-pass-valgrind|pretty|debug-info)"),
          optflag("", "ignored", "run tests marked as ignored"),
          optopt("", "runtool", "supervisor program to run tests under \
                                 (eg. emulator, valgrind)", "PROGRAM"),
          optopt("", "host-rustcflags", "flags to pass to rustc for host", "FLAGS"),
          optopt("", "target-rustcflags", "flags to pass to rustc for target", "FLAGS"),
          optflag("", "verbose", "run tests verbosely, showing all output"),
          optopt("", "logfile", "file to log test execution to", "FILE"),
          optflag("", "jit", "run tests under the JIT"),
          optopt("", "target", "the target to build for", "TARGET"),
          optopt("", "host", "the host to build for", "HOST"),
          optopt("", "gdb-version", "the version of GDB used", "VERSION STRING"),
          optopt("", "lldb-version", "the version of LLDB used", "VERSION STRING"),
          optopt("", "android-cross-path", "Android NDK standalone path", "PATH"),
          optopt("", "adb-path", "path to the android debugger", "PATH"),
          optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH"),
          optopt("", "lldb-python-dir", "directory containing LLDB's python module", "PATH"),
          optflag("h", "help", "show this message"));

    assert!(!args.is_empty());
    let argv0 = args[0].clone();
    let args_ = args.tail();
    if args[1] == "-h" || args[1] == "--help" {
        let message = format!("Usage: {} [OPTIONS] [TESTNAME...]", argv0);
        println!("{}", getopts::usage(&message, &groups));
        println!("");
        panic!()
    }

    let matches =
        &match getopts::getopts(args_, &groups) {
          Ok(m) => m,
          Err(f) => panic!("{:?}", f)
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        let message = format!("Usage: {} [OPTIONS]  [TESTNAME...]", argv0);
        println!("{}", getopts::usage(&message, &groups));
        println!("");
        panic!()
    }

    fn opt_path(m: &getopts::Matches, nm: &str) -> Path {
        match m.opt_str(nm) {
            Some(s) => Path::new(s),
            None => panic!("no option (=path) found for {}", nm),
        }
    }

    let filter = if !matches.free.is_empty() {
        Some(matches.free[0].clone())
    } else {
        None
    };

    Config {
        compile_lib_path: matches.opt_str("compile-lib-path").unwrap(),
        run_lib_path: matches.opt_str("run-lib-path").unwrap(),
        rustc_path: opt_path(matches, "rustc-path"),
        clang_path: matches.opt_str("clang-path").map(|s| Path::new(s)),
        valgrind_path: matches.opt_str("valgrind-path"),
        force_valgrind: matches.opt_present("force-valgrind"),
        llvm_bin_path: matches.opt_str("llvm-bin-path").map(|s| Path::new(s)),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        aux_base: opt_path(matches, "aux-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode: matches.opt_str("mode").unwrap().parse().ok().expect("invalid mode"),
        run_ignored: matches.opt_present("ignored"),
        filter: filter,
        logfile: matches.opt_str("logfile").map(|s| Path::new(s)),
        runtool: matches.opt_str("runtool"),
        host_rustcflags: matches.opt_str("host-rustcflags"),
        target_rustcflags: matches.opt_str("target-rustcflags"),
        jit: matches.opt_present("jit"),
        target: opt_str2(matches.opt_str("target")),
        host: opt_str2(matches.opt_str("host")),
        gdb_version: extract_gdb_version(matches.opt_str("gdb-version")),
        lldb_version: extract_lldb_version(matches.opt_str("lldb-version")),
        android_cross_path: opt_path(matches, "android-cross-path"),
        adb_path: opt_str2(matches.opt_str("adb-path")),
        adb_test_dir: format!("{}/{}",
            opt_str2(matches.opt_str("adb-test-dir")),
            opt_str2(matches.opt_str("target"))),
        adb_device_status:
            opt_str2(matches.opt_str("target")).contains("android") &&
            "(none)" != opt_str2(matches.opt_str("adb-test-dir")) &&
            !opt_str2(matches.opt_str("adb-test-dir")).is_empty(),
        lldb_python_dir: matches.opt_str("lldb-python-dir"),
        verbose: matches.opt_present("verbose"),
    }
}

pub fn log_config(config: &Config) {
    let c = config;
    logv(c, format!("configuration:"));
    logv(c, format!("compile_lib_path: {:?}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {:?}", config.run_lib_path));
    logv(c, format!("rustc_path: {:?}", config.rustc_path.display()));
    logv(c, format!("src_base: {:?}", config.src_base.display()));
    logv(c, format!("build_base: {:?}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", config.mode));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filter: {}",
                    opt_str(&config.filter
                                   .as_ref()
                                   .map(|re| re.to_string()))));
    logv(c, format!("runtool: {}", opt_str(&config.runtool)));
    logv(c, format!("host-rustcflags: {}",
                    opt_str(&config.host_rustcflags)));
    logv(c, format!("target-rustcflags: {}",
                    opt_str(&config.target_rustcflags)));
    logv(c, format!("jit: {}", config.jit));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("android-cross-path: {:?}",
                    config.android_cross_path.display()));
    logv(c, format!("adb_path: {:?}", config.adb_path));
    logv(c, format!("adb_test_dir: {:?}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}",
                    config.adb_device_status));
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("\n"));
}

pub fn opt_str<'a>(maybestr: &'a Option<String>) -> &'a str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => s,
    }
}

pub fn opt_str2(maybestr: Option<String>) -> String {
    match maybestr {
        None => "(none)".to_string(),
        Some(s) => s,
    }
}

pub fn run_tests(config: &Config) {
    if config.target.contains("android") {
        match config.mode {
            DebugInfoGdb => {
                println!("{} debug-info test uses tcp 5039 port.\
                         please reserve it", config.target);
            }
            _ =>{}
        }

        // android debug-info test uses remote debugger
        // so, we test 1 task at once.
        // also trying to isolate problems with adb_run_wrapper.sh ilooping
        env::set_var("RUST_TEST_TASKS","1");
    }

    match config.mode {
        DebugInfoLldb => {
            // Some older versions of LLDB seem to have problems with multiple
            // instances running in parallel, so only run one test task at a
            // time.
            env::set_var("RUST_TEST_TASKS", "1");
        }
        _ => { /* proceed */ }
    }

    let opts = test_opts(config);
    let tests = make_tests(config);
    // sadly osx needs some file descriptor limits raised for running tests in
    // parallel (especially when we have lots and lots of child processes).
    // For context, see #8904
    old_io::test::raise_fd_limit();
    // Prevent issue #21352 UAC blocking .exe containing 'patch' etc. on Windows
    // If #11207 is resolved (adding manifest to .exe) this becomes unnecessary
    env::set_var("__COMPAT_LAYER", "RunAsInvoker");
    let res = test::run_tests_console(&opts, tests.into_iter().collect());
    match res {
        Ok(true) => {}
        Ok(false) => panic!("Some tests failed"),
        Err(e) => {
            println!("I/O failure during tests: {:?}", e);
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
        nocapture: false,
        color: test::AutoColor,
    }
}

pub fn make_tests(config: &Config) -> Vec<test::TestDescAndFn> {
    debug!("making tests from {:?}",
           config.src_base.display());
    let mut tests = Vec::new();
    let dirs = fs::readdir(&config.src_base).unwrap();
    for file in &dirs {
        let file = file.clone();
        debug!("inspecting file {:?}", file.display());
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

    for ext in &valid_extensions {
        if name.ends_with(ext) {
            valid = true;
        }
    }

    for pre in &invalid_prefixes {
        if name.starts_with(pre) {
            valid = false;
        }
    }

    return valid;
}

pub fn make_test<F>(config: &Config, testfile: &Path, f: F) -> test::TestDescAndFn where
    F: FnOnce() -> test::TestFn,
{
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: make_test_name(config, testfile),
            ignore: header::is_test_ignored(config, testfile),
            should_fail: test::ShouldFail::No,
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
    test::DynTestFn(Thunk::new(move || {
        runtest::run(config, testfile)
    }))
}

pub fn make_metrics_test_closure(config: &Config, testfile: &Path) -> test::TestFn {
    let config = (*config).clone();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let testfile = testfile.as_str().unwrap().to_string();
    test::DynMetricFn(box move |mm: &mut test::MetricMap| {
        runtest::run_metrics(config, testfile, mm)
    })
}

fn extract_gdb_version(full_version_line: Option<String>) -> Option<String> {
    match full_version_line {
        Some(ref full_version_line)
          if full_version_line.trim().len() > 0 => {
            let full_version_line = full_version_line.trim();

            // used to be a regex "(^|[^0-9])([0-9]\.[0-9])([^0-9]|$)"
            for (pos, c) in full_version_line.char_indices() {
                if !c.is_digit(10) { continue }
                if pos + 2 >= full_version_line.len() { continue }
                if full_version_line.char_at(pos + 1) != '.' { continue }
                if !full_version_line.char_at(pos + 2).is_digit(10) { continue }
                if pos > 0 && full_version_line.char_at_reverse(pos).is_digit(10) {
                    continue
                }
                if pos + 3 < full_version_line.len() &&
                   full_version_line.char_at(pos + 3).is_digit(10) {
                    continue
                }
                return Some(full_version_line[pos..pos+3].to_string());
            }
            println!("Could not extract GDB version from line '{}'",
                     full_version_line);
            None
        },
        _ => None
    }
}

fn extract_lldb_version(full_version_line: Option<String>) -> Option<String> {
    // Extract the major LLDB version from the given version string.
    // LLDB version strings are different for Apple and non-Apple platforms.
    // At the moment, this function only supports the Apple variant, which looks
    // like this:
    //
    // LLDB-179.5 (older versions)
    // lldb-300.2.51 (new versions)
    //
    // We are only interested in the major version number, so this function
    // will return `Some("179")` and `Some("300")` respectively.

    match full_version_line {
        Some(ref full_version_line)
          if full_version_line.trim().len() > 0 => {
            let full_version_line = full_version_line.trim();

            for (pos, l) in full_version_line.char_indices() {
                if l != 'l' && l != 'L' { continue }
                if pos + 5 >= full_version_line.len() { continue }
                let l = full_version_line.char_at(pos + 1);
                if l != 'l' && l != 'L' { continue }
                let d = full_version_line.char_at(pos + 2);
                if d != 'd' && d != 'D' { continue }
                let b = full_version_line.char_at(pos + 3);
                if b != 'b' && b != 'B' { continue }
                let dash = full_version_line.char_at(pos + 4);
                if dash != '-' { continue }

                let vers = full_version_line[pos + 5..].chars().take_while(|c| {
                    c.is_digit(10)
                }).collect::<String>();
                if vers.len() > 0 { return Some(vers) }
            }
            println!("Could not extract LLDB version from line '{}'",
                     full_version_line);
            None
        },
        _ => None
    }
}
