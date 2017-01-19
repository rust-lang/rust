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
#![feature(static_in_const)]
#![feature(test)]
#![feature(libc)]

#![deny(warnings)]

extern crate libc;
extern crate test;
extern crate getopts;

#[cfg(cargobuild)]
extern crate rustc_serialize;
#[cfg(not(cargobuild))]
extern crate serialize as rustc_serialize;

#[macro_use]
extern crate log;

#[cfg(cargobuild)]
extern crate env_logger;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use getopts::{optopt, optflag, reqopt};
use common::Config;
use common::{Pretty, DebugInfoGdb, DebugInfoLldb, Mode};
use test::TestPaths;
use util::logv;

use self::header::EarlyProps;

pub mod procsrv;
pub mod util;
mod json;
pub mod header;
pub mod runtest;
pub mod common;
pub mod errors;
mod raise_fd_limit;
mod uidiff;

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
        vec![reqopt("", "compile-lib-path", "path to host shared libraries", "PATH"),
          reqopt("", "run-lib-path", "path to target shared libraries", "PATH"),
          reqopt("", "rustc-path", "path to rustc to use for compiling", "PATH"),
          reqopt("", "rustdoc-path", "path to rustdoc to use for compiling", "PATH"),
          reqopt("", "lldb-python", "path to python to use for doc tests", "PATH"),
          reqopt("", "docck-python", "path to python to use for doc tests", "PATH"),
          optopt("", "valgrind-path", "path to Valgrind executable for Valgrind tests", "PROGRAM"),
          optflag("", "force-valgrind", "fail if Valgrind tests cannot be run under Valgrind"),
          optopt("", "llvm-filecheck", "path to LLVM's FileCheck binary", "DIR"),
          reqopt("", "src-base", "directory to scan for test files", "PATH"),
          reqopt("", "build-base", "directory to deposit test outputs", "PATH"),
          reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET"),
          reqopt("", "mode", "which sort of compile tests to run",
                 "(compile-fail|parse-fail|run-fail|run-pass|\
                  run-pass-valgrind|pretty|debug-info|incremental|mir-opt)"),
          optflag("", "ignored", "run tests marked as ignored"),
          optflag("", "exact", "filters match exactly"),
          optopt("", "runtool", "supervisor program to run tests under \
                                 (eg. emulator, valgrind)", "PROGRAM"),
          optopt("", "host-rustcflags", "flags to pass to rustc for host", "FLAGS"),
          optopt("", "target-rustcflags", "flags to pass to rustc for target", "FLAGS"),
          optflag("", "verbose", "run tests verbosely, showing all output"),
          optflag("", "quiet", "print one character per test instead of one line"),
          optopt("", "logfile", "file to log test execution to", "FILE"),
          optopt("", "target", "the target to build for", "TARGET"),
          optopt("", "host", "the host to build for", "HOST"),
          optopt("", "gdb", "path to GDB to use for GDB debuginfo tests", "PATH"),
          optopt("", "lldb-version", "the version of LLDB used", "VERSION STRING"),
          optopt("", "llvm-version", "the version of LLVM used", "VERSION STRING"),
          optopt("", "android-cross-path", "Android NDK standalone path", "PATH"),
          optopt("", "adb-path", "path to the android debugger", "PATH"),
          optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH"),
          optopt("", "lldb-python-dir", "directory containing LLDB's python module", "PATH"),
          reqopt("", "cc", "path to a C compiler", "PATH"),
          reqopt("", "cxx", "path to a C++ compiler", "PATH"),
          reqopt("", "cflags", "flags for the C compiler", "FLAGS"),
          reqopt("", "llvm-components", "list of LLVM components built in", "LIST"),
          reqopt("", "llvm-cxxflags", "C++ flags for LLVM", "FLAGS"),
          optopt("", "nodejs", "the name of nodejs", "PATH"),
          optflag("h", "help", "show this message")];

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

    let (gdb, gdb_version, gdb_native_rust) = analyze_gdb(matches.opt_str("gdb"));

    Config {
        compile_lib_path: make_absolute(opt_path(matches, "compile-lib-path")),
        run_lib_path: make_absolute(opt_path(matches, "run-lib-path")),
        rustc_path: opt_path(matches, "rustc-path"),
        rustdoc_path: opt_path(matches, "rustdoc-path"),
        lldb_python: matches.opt_str("lldb-python").unwrap(),
        docck_python: matches.opt_str("docck-python").unwrap(),
        valgrind_path: matches.opt_str("valgrind-path"),
        force_valgrind: matches.opt_present("force-valgrind"),
        llvm_filecheck: matches.opt_str("llvm-filecheck").map(|s| PathBuf::from(&s)),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode: matches.opt_str("mode").unwrap().parse().ok().expect("invalid mode"),
        run_ignored: matches.opt_present("ignored"),
        filter: matches.free.first().cloned(),
        filter_exact: matches.opt_present("exact"),
        logfile: matches.opt_str("logfile").map(|s| PathBuf::from(&s)),
        runtool: matches.opt_str("runtool"),
        host_rustcflags: matches.opt_str("host-rustcflags"),
        target_rustcflags: matches.opt_str("target-rustcflags"),
        target: opt_str2(matches.opt_str("target")),
        host: opt_str2(matches.opt_str("host")),
        gdb: gdb,
        gdb_version: gdb_version,
        gdb_native_rust: gdb_native_rust,
        lldb_version: extract_lldb_version(matches.opt_str("lldb-version")),
        llvm_version: matches.opt_str("llvm-version"),
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

        cc: matches.opt_str("cc").unwrap(),
        cxx: matches.opt_str("cxx").unwrap(),
        cflags: matches.opt_str("cflags").unwrap(),
        llvm_components: matches.opt_str("llvm-components").unwrap(),
        llvm_cxxflags: matches.opt_str("llvm-cxxflags").unwrap(),
        nodejs: matches.opt_str("nodejs"),
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
    logv(c, format!("filter_exact: {}", config.filter_exact));
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
        match config.mode {
            // These tests don't actually run code or don't run for android, so
            // we don't need to limit ourselves there
            Mode::Ui |
            Mode::CompileFail |
            Mode::ParseFail |
            Mode::RunMake |
            Mode::Codegen |
            Mode::CodegenUnits |
            Mode::Pretty |
            Mode::Rustdoc => {}

            _ => {
                env::set_var("RUST_TEST_THREADS", "1");
            }

        }
    }

    match config.mode {
        DebugInfoLldb => {
            if let Some(lldb_version) = config.lldb_version.as_ref() {
                if is_blacklisted_lldb_version(&lldb_version[..]) {
                    println!("WARNING: The used version of LLDB ({}) has a \
                              known issue that breaks debuginfo tests. See \
                              issue #32520 for more information. Skipping all \
                              LLDB-based tests!",
                             lldb_version);
                    return
                }
            }

            // Some older versions of LLDB seem to have problems with multiple
            // instances running in parallel, so only run one test thread at a
            // time.
            env::set_var("RUST_TEST_THREADS", "1");
        }
        _ => { /* proceed */ }
    }

    // FIXME(#33435) Avoid spurious failures in codegen-units/partitioning tests.
    if let Mode::CodegenUnits = config.mode {
        let _ = fs::remove_dir_all("tmp/partitioning-tests");
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

    // Let tests know which target they're running as
    env::set_var("TARGET", &config.target);

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
        filter_exact: config.filter_exact,
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
        test_threads: None,
        skip: vec![],
        list: false,
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
        let name = file.file_name();
        if name == *"compiletest-ignore-dir" {
            return Ok(());
        }
        if name == *"Makefile" && config.mode == Mode::RunMake {
            let paths = TestPaths {
                file: dir.to_path_buf(),
                base: base.to_path_buf(),
                relative_dir: relative_dir_path.parent().unwrap().to_path_buf(),
            };
            tests.push(make_test(config, &paths));
            return Ok(())
        }
    }

    // If we find a test foo/bar.rs, we have to build the
    // output directory `$build/foo` so we can write
    // `$build/foo/bar` into it. We do this *now* in this
    // sequential loop because otherwise, if we do it in the
    // tests themselves, they race for the privilege of
    // creating the directories and sometimes fail randomly.
    let build_dir = config.build_base.join(&relative_dir_path);
    fs::create_dir_all(&build_dir).unwrap();

    // Add each `.rs` file as a test, and recurse further on any
    // subdirectories we find, except for `aux` directories.
    let dirs = fs::read_dir(dir)?;
    for file in dirs {
        let file = file?;
        let file_path = file.path();
        let file_name = file.file_name();
        if is_test(&file_name) {
            debug!("found test file: {:?}", file_path.display());
            let paths = TestPaths {
                file: file_path,
                base: base.to_path_buf(),
                relative_dir: relative_dir_path.to_path_buf(),
            };
            tests.push(make_test(config, &paths))
        } else if file_path.is_dir() {
            let relative_file_path = relative_dir_path.join(file.file_name());
            if &file_name == "auxiliary" {
                // `aux` directories contain other crates used for
                // cross-crate tests. Don't search them for tests, but
                // do create a directory in the build dir for them,
                // since we will dump intermediate output in there
                // sometimes.
                let build_dir = config.build_base.join(&relative_file_path);
                fs::create_dir_all(&build_dir).unwrap();
            } else {
                debug!("found directory: {:?}", file_path.display());
                collect_tests_from_dir(config,
                                       base,
                                       &file_path,
                                       &relative_file_path,
                                       tests)?;
            }
        } else {
            debug!("found other file/directory: {:?}", file_path.display());
        }
    }
    Ok(())
}

pub fn is_test(file_name: &OsString) -> bool {
    let file_name = file_name.to_str().unwrap();

    if !file_name.ends_with(".rs") {
        return false;
    }

    // `.`, `#`, and `~` are common temp-file prefixes.
    let invalid_prefixes = &[".", "#", "~"];
    !invalid_prefixes.iter().any(|p| file_name.starts_with(p))
}

pub fn make_test(config: &Config, testpaths: &TestPaths) -> test::TestDescAndFn {
    let early_props = EarlyProps::from_file(config, &testpaths.file);

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

    // Debugging emscripten code doesn't make sense today
    let mut ignore = early_props.ignore;
    if (config.mode == DebugInfoGdb || config.mode == DebugInfoLldb) &&
        config.target.contains("emscripten") {
        ignore = true;
    }

    test::TestDescAndFn {
        desc: test::TestDesc {
            name: make_test_name(config, testpaths),
            ignore: ignore,
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
    test::DynTestFn(Box::new(move |()| {
        runtest::run(config, &testpaths)
    }))
}

/// Returns (Path to GDB, GDB Version, GDB has Rust Support)
fn analyze_gdb(gdb: Option<String>) -> (Option<String>, Option<u32>, bool) {
    #[cfg(not(windows))]
    const GDB_FALLBACK: &str = "gdb";
    #[cfg(windows)]
    const GDB_FALLBACK: &str = "gdb.exe";

    const MIN_GDB_WITH_RUST: u32 = 7011010;

    let gdb = match gdb {
        None => GDB_FALLBACK,
        Some(ref s) if s.is_empty() => GDB_FALLBACK, // may be empty if configure found no gdb
        Some(ref s) => s,
    };

    let version_line = Command::new(gdb).arg("--version").output().map(|output| {
        String::from_utf8_lossy(&output.stdout).lines().next().unwrap().to_string()
    }).ok();

    let version = match version_line {
        Some(line) => extract_gdb_version(&line),
        None => return (None, None, false),
    };

    let gdb_native_rust = version.map_or(false, |v| v >= MIN_GDB_WITH_RUST);

    return (Some(gdb.to_owned()), version, gdb_native_rust);
}

fn extract_gdb_version(full_version_line: &str) -> Option<u32> {
    let full_version_line = full_version_line.trim();

    // GDB versions look like this: "major.minor.patch?.yyyymmdd?", with both
    // of the ? sections being optional

    // We will parse up to 3 digits for minor and patch, ignoring the date
    // We limit major to 1 digit, otherwise, on openSUSE, we parse the openSUSE version

    // don't start parsing in the middle of a number
    let mut prev_was_digit = false;
    for (pos, c) in full_version_line.char_indices() {
        if prev_was_digit || !c.is_digit(10) {
            prev_was_digit = c.is_digit(10);
            continue
        }

        prev_was_digit = true;

        let line = &full_version_line[pos..];

        let next_split = match line.find(|c: char| !c.is_digit(10)) {
            Some(idx) => idx,
            None => continue, // no minor version
        };

        if line.as_bytes()[next_split] != b'.' {
            continue; // no minor version
        }

        let major = &line[..next_split];
        let line = &line[next_split + 1..];

        let (minor, patch) = match line.find(|c: char| !c.is_digit(10)) {
            Some(idx) => if line.as_bytes()[idx] == b'.' {
                let patch = &line[idx + 1..];

                let patch_len = patch.find(|c: char| !c.is_digit(10)).unwrap_or(patch.len());
                let patch = &patch[..patch_len];
                let patch = if patch_len > 3 || patch_len == 0 { None } else { Some(patch) };

                (&line[..idx], patch)
            } else {
                (&line[..idx], None)
            },
            None => (line, None),
        };

        if major.len() != 1 || minor.is_empty() {
            continue;
        }

        let major: u32 = major.parse().unwrap();
        let minor: u32 = minor.parse().unwrap();
        let patch: u32 = patch.unwrap_or("0").parse().unwrap();

        return Some(((major * 1000) + minor) * 1000 + patch);
    }

    None
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
        }
    }
    None
}

fn is_blacklisted_lldb_version(version: &str) -> bool {
    version == "350"
}

#[test]
fn test_extract_gdb_version() {
    macro_rules! test { ($($expectation:tt: $input:tt,)*) => {{$(
        assert_eq!(extract_gdb_version($input), Some($expectation));
    )*}}}

    test! {
        7000001: "GNU gdb (GDB) CentOS (7.0.1-45.el5.centos)",

        7002000: "GNU gdb (GDB) Red Hat Enterprise Linux (7.2-90.el6)",

        7004000: "GNU gdb (Ubuntu/Linaro 7.4-2012.04-0ubuntu2.1) 7.4-2012.04",
        7004001: "GNU gdb (GDB) 7.4.1-debian",

        7006001: "GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-80.el7",

        7007001: "GNU gdb (Ubuntu 7.7.1-0ubuntu5~14.04.2) 7.7.1",
        7007001: "GNU gdb (Debian 7.7.1+dfsg-5) 7.7.1",
        7007001: "GNU gdb (GDB) Fedora 7.7.1-21.fc20",

        7008000: "GNU gdb (GDB; openSUSE 13.2) 7.8",
        7009001: "GNU gdb (GDB) Fedora 7.9.1-20.fc22",
        7010001: "GNU gdb (GDB) Fedora 7.10.1-31.fc23",

        7011000: "GNU gdb (Ubuntu 7.11-0ubuntu1) 7.11",
        7011001: "GNU gdb (Ubuntu 7.11.1-0ubuntu1~16.04) 7.11.1",
        7011001: "GNU gdb (Debian 7.11.1-2) 7.11.1",
        7011001: "GNU gdb (GDB) Fedora 7.11.1-86.fc24",
        7011001: "GNU gdb (GDB; openSUSE Leap 42.1) 7.11.1",
        7011001: "GNU gdb (GDB; openSUSE Tumbleweed) 7.11.1",

        7011090: "7.11.90",
        7011090: "GNU gdb (Ubuntu 7.11.90.20161005-0ubuntu1) 7.11.90.20161005-git",

        7012000: "7.12",
        7012000: "GNU gdb (GDB) 7.12",
        7012000: "GNU gdb (GDB) 7.12.20161027-git",
        7012050: "GNU gdb (GDB) 7.12.50.20161027-git",
    }
}
