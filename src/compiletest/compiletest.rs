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

use extra::getopts;
use extra::test;

use common::config;
use common::mode_run_pass;
use common::mode_run_fail;
use common::mode_compile_fail;
use common::mode_pretty;
use common::mode_debug_info;
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
    let opts =
        ~[getopts::reqopt("compile-lib-path"),
          getopts::reqopt("run-lib-path"),
          getopts::reqopt("rustc-path"), getopts::reqopt("src-base"),
          getopts::reqopt("build-base"), getopts::reqopt("aux-base"),
          getopts::reqopt("stage-id"),
          getopts::reqopt("mode"), getopts::optflag("ignored"),
          getopts::optopt("runtool"), getopts::optopt("rustcflags"),
          getopts::optflag("verbose"),
          getopts::optopt("logfile"),
          getopts::optflag("jit"),
          getopts::optflag("newrt"),
          getopts::optopt("target"),
          getopts::optopt("adb-path"),
          getopts::optopt("adb-test-dir")
         ];

    assert!(!args.is_empty());
    let args_ = args.tail();
    let matches =
        &match getopts::getopts(args_, opts) {
          Ok(m) => m,
          Err(f) => fail!(getopts::fail_str(f))
        };

    fn opt_path(m: &getopts::Matches, nm: &str) -> Path {
        Path(getopts::opt_str(m, nm))
    }

    config {
        compile_lib_path: getopts::opt_str(matches, "compile-lib-path"),
        run_lib_path: getopts::opt_str(matches, "run-lib-path"),
        rustc_path: opt_path(matches, "rustc-path"),
        src_base: opt_path(matches, "src-base"),
        build_base: opt_path(matches, "build-base"),
        aux_base: opt_path(matches, "aux-base"),
        stage_id: getopts::opt_str(matches, "stage-id"),
        mode: str_mode(getopts::opt_str(matches, "mode")),
        run_ignored: getopts::opt_present(matches, "ignored"),
        filter:
             if !matches.free.is_empty() {
                 Some(copy matches.free[0])
             } else { None },
        logfile: getopts::opt_maybe_str(matches, "logfile").map(|s| Path(*s)),
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
        filter: copy config.filter,
        run_ignored: config.run_ignored,
        logfile: copy config.logfile,
        run_tests: true,
        run_benchmarks: false,
        save_results: None,
        compare_results: None
    }
}

pub fn make_tests(config: &config) -> ~[test::TestDescAndFn] {
    debug!("making tests from %s",
           config.src_base.to_str());
    let mut tests = ~[];
    let dirs = os::list_dir_path(&config.src_base);
    for dirs.iter().advance |file| {
        let file = copy *file;
        debug!("inspecting file %s", file.to_str());
        if is_test(config, file) {
            tests.push(make_test(config, file))
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

pub fn make_test(config: &config, testfile: &Path) -> test::TestDescAndFn {
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: make_test_name(config, testfile),
            ignore: header::is_test_ignored(config, testfile),
            should_fail: false
        },
        testfn: make_test_closure(config, testfile),
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
    let config = Cell::new(copy *config);
    let testfile = Cell::new(testfile.to_str());
    test::DynTestFn(|| { runtest::run(config.take(), testfile.take()) })
}
