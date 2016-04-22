// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "compiletest"]

#![feature(box_syntax)]
#![feature(rustc_private)]
#![feature(test)]
#![feature(question_mark)]
#![feature(libc)]

#![deny(warnings)]

extern crate libc;
extern crate test;
extern crate getopts;
extern crate serialize as rustc_serialize;

#[macro_use]
extern crate log;

#[cfg(cargobuild)]
extern crate env_logger;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use getopts::{optopt, optflag, reqopt};
use common::Config;
use common::{Pretty, DebugInfoGdb, DebugInfoLldb};
use test::TestPaths;
use util::logv;

pub mod procsrv;
pub mod util;
mod json;
pub mod header;
pub mod runtest;
pub mod common;
pub mod errors;
mod raise_fd_limit;

fn main() {
    #[cfg(cargobuild)]
    fn log_init() { env_logger::init().unwrap(); }
    #[cfg(not(cargobuild))]
    fn log_init() {}
    log_init();

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
          reqopt("", "rustdoc-path", "path to rustdoc to use for compiling", "PATH"),
          reqopt("", "python", "path to python to use for doc tests", "PATH"),
          optopt("", "valgrind-path", "path to Valgrind executable for Valgrind tests", "PROGRAM"),
          optflag("", "force-valgrind", "fail if Valgrind tests cannot be run under Valgrind"),
          optopt("", "llvm-filecheck", "path to LLVM's FileCheck binary", "DIR"),
          reqopt("", "src-base", "directory to scan for test files", "PATH"),
          reqopt("", "build-base", "directory to deposit test outputs", "PATH"),
          reqopt("", "aux-base", "directory to find auxiliary test files", "PATH"),
          reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET"),
          reqopt("", "mode", "which sort of compile tests to run",
                 "(compile-fail|parse-fail|run-fail|run-pass|\
                  run-pass-valgrind|pretty|debug-info|incremental)"),
          optflag("", "ignored", "run tests marked as ignored"),
          optopt("", "runtool", "supervisor program to run tests under \
                                 (eg. emulator, valgrind)", "PROGRAM"),
          optopt("", "host-rustcflags", "flags to pass to rustc for host", "FLAGS"),
          optopt("", "target-rustcflags", "flags to pass to rustc for target", "FLAGS"),
          optflag("", "verbose", "run tests verbosely, showing all output"),
          optflag("", "quiet", "print one character per test instead of one line"),
          optopt("", "logfile", "file to log test execution to", "FILE"),
          optopt("", "target", "the target to build for", "TARGET"),
          optopt("", "host", "the host to build for", "HOST"),
          optopt("", "gdb-version", "the version of GDB used", "VERSION STRING"),
          optopt("", "lldb-version", "the version of LLDB used", "VERSION STRING"),
          optopt("", "android-cross-path", "Android NDK standalone path", "PATH"),
          optopt("", "adb-path", "path to the android debugger", "PATH"),
          optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH"),
          optopt("", "lldb-python-dir", "directory containing LLDB's python module", "PATH"),
          optflag("h", "help", "show this message"));

    let (argv0, args_) = args.split_first().unwrap();
    if args.len() == 1 || args[1] == "-h" || args[1] == "--help" {
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

    fn opt_path(m: &getopts::Matches, nm: &str) -> PathBuf {
        match m.opt_str(nm) {
            Some(s) => PathBuf::from(&s),
            None => panic!("no option (=path) found for {}", nm),
        }
    }

    fn make_absolute(path: PathBuf) -> PathBuf {
        if path.is_relative() {
            env::current_dir().unwrap().join(path)
        } else {
            path
        }
    }

    Config {
        compile_lib_path: make_absolute(opt_path(matches, "compile-lib-path")),
        run_lib_path: make_absolute(opt_path(matches, "run-lib-path")),
        rustc_path: opt_path(matches, "rustc-path"),
        rustdoc_path: opt_path(matches, "rustdoc-path"),
        python: matches.opt_str("python").unwrap(),
        valgrind_path: matches.opt_str("valgrind-path"),
        force_valgrind: matches.opt_present("force-valgrind"),
        llvm_filecheck: matches.opt_str("llvm-filecheck").map(|s| PathBuf::from(&s)),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        aux_base: opt_path(matches, "aux-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode: matches.opt_str("mode").unwrap().parse().ok().expect("invalid mode"),
        run_ignored: matches.opt_present("ignored"),
        filter: matches.free.first().cloned(),
        logfile: matches.opt_str("logfile").map(|s| PathBuf::from(&s)),
        runtool: matches.opt_str("runtool"),
        host_rustcflags: matches.opt_str("host-rustcflags"),
        target_rustcflags: matches.opt_str("target-rustcflags"),
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
        quiet: matches.opt_present("quiet"),
    }
}

pub fn log_config(config: &Config) {
    let c = config;
    logv(c, format!("configuration:"));
    logv(c, format!("compile_lib_path: {:?}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {:?}", config.run_lib_path));
    logv(c, format!("rustc_path: {:?}", config.rustc_path.display()));
    logv(c, format!("rustdoc_path: {:?}", config.rustdoc_path.display()));
    logv(c, format!("src_base: {:?}", config.src_base.display()));
    logv(c, format!("build_base: {:?}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", config.mode));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filter: {}",
                    opt_str(&config.filter
                                   .as_ref()
                                   .map(|re| re.to_owned()))));
    logv(c, format!("runtool: {}", opt_str(&config.runtool)));
    logv(c, format!("host-rustcflags: {}",
                    opt_str(&config.host_rustcflags)));
    logv(c, format!("target-rustcflags: {}",
                    opt_str(&config.target_rustcflags)));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("android-cross-path: {:?}",
                    config.android_cross_path.display()));
    logv(c, format!("adb_path: {:?}", config.adb_path));
    logv(c, format!("adb_test_dir: {:?}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}",
                    config.adb_device_status));
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("quiet: {}", config.quiet));
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
        None => "(none)".to_owned(),
        Some(s) => s,
    }
}

pub fn run_tests(config: &Config) {
    if config.target.contains("android") {
        if let DebugInfoGdb = config.mode {
            println!("{} debug-info test uses tcp 5039 port.\
                     please reserve it", config.target);
        }

        // android debug-info test uses remote debugger
        // so, we test 1 thread at once.
        // also trying to isolate problems with adb_run_wrapper.sh ilooping
        env::set_var("RUST_TEST_THREADS","1");
    }

    match config.mode {
        DebugInfoLldb => {
            // Some older versions of LLDB seem to have problems with multiple
            // instances running in parallel, so only run one test thread at a
            // time.
            env::set_var("RUST_TEST_THREADS", "1");
        }
        _ => { /* proceed */ }
    }

    let opts = test_opts(config);
    let tests = make_tests(config);
    // sadly osx needs some file descriptor limits raised for running tests in
    // parallel (especially when we have lots and lots of child processes).
    // For context, see #8904
    unsafe { raise_fd_limit::raise_fd_limit(); }
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
        filter: config.filter.clone(),
        run_ignored: config.run_ignored,
        quiet: config.quiet,
        logfile: config.logfile.clone(),
        run_tests: true,
        bench_benchmarks: true,
        nocapture: match env::var("RUST_TEST_NOCAPTURE") {
            Ok(val) => &val != "0",
            Err(_) => false
        },
        color: test::AutoColor,
    }
}

pub fn make_tests(config: &Config) -> Vec<test::TestDescAndFn> {
    debug!("making tests from {:?}",
           config.src_base.display());
    let mut tests = Vec::new();
    collect_tests_from_dir(config,
                           &config.src_base,
                           &config.src_base,
                           &PathBuf::new(),
                           &mut tests)
        .unwrap();
    tests
}

fn collect_tests_from_dir(config: &Config,
                          base: &Path,
                          dir: &Path,
                          relative_dir_path: &Path,
                          tests: &mut Vec<test::TestDescAndFn>)
                          -> io::Result<()> {
    // Ignore directories that contain a file
    // `compiletest-ignore-dir`.
    for file in fs::read_dir(dir)? {
        let file = file?;
        if file.file_name() == *"compiletest-ignore-dir" {
            return Ok(());
        }
    }

    let dirs = fs::read_dir(dir)?;
    for file in dirs {
        let file = file?;
        let file_path = file.path();
        debug!("inspecting file {:?}", file_path.display());
        if is_test(config, &file_path) {
            // If we find a test foo/bar.rs, we have to build the
            // output directory `$build/foo` so we can write
            // `$build/foo/bar` into it. We do this *now* in this
            // sequential loop because otherwise, if we do it in the
            // tests themselves, they race for the privilege of
            // creating the directories and sometimes fail randomly.
            let build_dir = config.build_base.join(&relative_dir_path);
            fs::create_dir_all(&build_dir).unwrap();

            let paths = TestPaths {
                file: file_path,
                base: base.to_path_buf(),
                relative_dir: relative_dir_path.to_path_buf(),
            };
            tests.push(make_test(config, &paths))
        } else if file_path.is_dir() {
            let relative_file_path = relative_dir_path.join(file.file_name());
            collect_tests_from_dir(config,
                                   base,
                                   &file_path,
                                   &relative_file_path,
                                   tests)?;
        }
    }
    Ok(())
}

pub fn is_test(config: &Config, testfile: &Path) -> bool {
    // Pretty-printer does not work with .rc files yet
    let valid_extensions =
        match config.mode {
          Pretty => vec!(".rs".to_owned()),
          _ => vec!(".rc".to_owned(), ".rs".to_owned())
        };
    let invalid_prefixes = vec!(".".to_owned(), "#".to_owned(), "~".to_owned());
    let name = testfile.file_name().unwrap().to_str().unwrap();

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

pub fn make_test(config: &Config, testpaths: &TestPaths) -> test::TestDescAndFn {
    let early_props = header::early_props(config, &testpaths.file);

    // The `should-fail` annotation doesn't apply to pretty tests,
    // since we run the pretty printer across all tests by default.
    // If desired, we could add a `should-fail-pretty` annotation.
    let should_panic = match config.mode {
        Pretty => test::ShouldPanic::No,
        _ => if early_props.should_fail {
            test::ShouldPanic::Yes
        } else {
            test::ShouldPanic::No
        }
    };

    test::TestDescAndFn {
        desc: test::TestDesc {
            name: make_test_name(config, testpaths),
            ignore: early_props.ignore,
            should_panic: should_panic,
        },
        testfn: make_test_closure(config, testpaths),
    }
}

pub fn make_test_name(config: &Config, testpaths: &TestPaths) -> test::TestName {
    // Convert a complete path to something like
    //
    //    run-pass/foo/bar/baz.rs
    let path =
        PathBuf::from(config.mode.to_string())
        .join(&testpaths.relative_dir)
        .join(&testpaths.file.file_name().unwrap());
    test::DynTestName(format!("[{}] {}", config.mode, path.display()))
}

pub fn make_test_closure(config: &Config, testpaths: &TestPaths) -> test::TestFn {
    let config = config.clone();
    let testpaths = testpaths.clone();
    test::DynTestFn(Box::new(move || {
        runtest::run(config, &testpaths)
    }))
}

fn extract_gdb_version(full_version_line: Option<String>) -> Option<String> {
    match full_version_line {
        Some(ref full_version_line)
          if !full_version_line.trim().is_empty() => {
            let full_version_line = full_version_line.trim();

            // used to be a regex "(^|[^0-9])([0-9]\.[0-9]+)"
            for (pos, c) in full_version_line.char_indices() {
                if !c.is_digit(10) {
                    continue
                }
                if pos + 2 >= full_version_line.len() {
                    continue
                }
                if full_version_line[pos + 1..].chars().next().unwrap() != '.' {
                    continue
                }
                if !full_version_line[pos + 2..].chars().next().unwrap().is_digit(10) {
                    continue
                }
                if pos > 0 && full_version_line[..pos].chars().next_back()
                                                      .unwrap().is_digit(10) {
                    continue
                }
                let mut end = pos + 3;
                while end < full_version_line.len() &&
                      full_version_line[end..].chars().next()
                                              .unwrap().is_digit(10) {
                    end += 1;
                }
                return Some(full_version_line[pos..end].to_owned());
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

    if let Some(ref full_version_line) = full_version_line {
        if !full_version_line.trim().is_empty() {
            let full_version_line = full_version_line.trim();

            for (pos, l) in full_version_line.char_indices() {
                if l != 'l' && l != 'L' { continue }
                if pos + 5 >= full_version_line.len() { continue }
                let l = full_version_line[pos + 1..].chars().next().unwrap();
                if l != 'l' && l != 'L' { continue }
                let d = full_version_line[pos + 2..].chars().next().unwrap();
                if d != 'd' && d != 'D' { continue }
                let b = full_version_line[pos + 3..].chars().next().unwrap();
                if b != 'b' && b != 'B' { continue }
                let dash = full_version_line[pos + 4..].chars().next().unwrap();
                if dash != '-' { continue }

                let vers = full_version_line[pos + 5..].chars().take_while(|c| {
                    c.is_digit(10)
                }).collect::<String>();
                if !vers.is_empty() { return Some(vers) }
            }
            println!("Could not extract LLDB version from line '{}'",
                     full_version_line);
        }
    }
    None
}
