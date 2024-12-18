#![crate_name = "compiletest"]
// The `test` crate is the only unstable feature
// allowed here, just to share similar code.
#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests;

pub mod common;
pub mod compute_diff;
mod debuggers;
pub mod errors;
pub mod header;
mod json;
mod raise_fd_limit;
mod read2;
pub mod runtest;
pub mod util;

use core::panic;
use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use std::{env, fs, vec};

use build_helper::git::{get_git_modified_files, get_git_untracked_files};
use getopts::Options;
use test::ColorConfig;
use tracing::*;
use walkdir::WalkDir;

use self::header::{EarlyProps, make_test_description};
use crate::common::{
    CompareMode, Config, Mode, PassMode, TestPaths, UI_EXTENSIONS, expected_output_path,
    output_base_dir, output_relative_path,
};
use crate::header::HeadersCache;
use crate::util::logv;

/// Creates the `Config` instance for this invocation of compiletest.
///
/// The config mostly reflects command-line arguments, but there might also be
/// some code here that inspects environment variables or even runs executables
/// (e.g. when discovering debugger versions).
pub fn parse_config(args: Vec<String>) -> Config {
    let mut opts = Options::new();
    opts.reqopt("", "compile-lib-path", "path to host shared libraries", "PATH")
        .reqopt("", "run-lib-path", "path to target shared libraries", "PATH")
        .reqopt("", "rustc-path", "path to rustc to use for compiling", "PATH")
        .optopt("", "cargo-path", "path to cargo to use for compiling", "PATH")
        .optopt("", "rustdoc-path", "path to rustdoc to use for compiling", "PATH")
        .optopt("", "coverage-dump-path", "path to coverage-dump to use in tests", "PATH")
        .reqopt("", "python", "path to python to use for doc tests", "PATH")
        .optopt("", "jsondocck-path", "path to jsondocck to use for doc tests", "PATH")
        .optopt("", "jsondoclint-path", "path to jsondoclint to use for doc tests", "PATH")
        .optopt("", "run-clang-based-tests-with", "path to Clang executable", "PATH")
        .optopt("", "llvm-filecheck", "path to LLVM's FileCheck binary", "DIR")
        .reqopt("", "src-base", "directory to scan for test files", "PATH")
        .reqopt("", "build-base", "directory to deposit test outputs", "PATH")
        .reqopt("", "sysroot-base", "directory containing the compiler sysroot", "PATH")
        .reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET")
        .reqopt(
            "",
            "mode",
            "which sort of compile tests to run",
            "pretty | debug-info | codegen | rustdoc \
            | rustdoc-json | codegen-units | incremental | run-make | ui \
            | js-doc-test | mir-opt | assembly | crashes",
        )
        .reqopt(
            "",
            "suite",
            "which suite of compile tests to run. used for nicer error reporting.",
            "SUITE",
        )
        .optopt(
            "",
            "pass",
            "force {check,build,run}-pass tests to this mode.",
            "check | build | run",
        )
        .optopt("", "run", "whether to execute run-* tests", "auto | always | never")
        .optflag("", "ignored", "run tests marked as ignored")
        .optflag("", "has-enzyme", "run tests that require enzyme")
        .optflag("", "with-rustc-debug-assertions", "whether rustc was built with debug assertions")
        .optflag("", "with-std-debug-assertions", "whether std was built with debug assertions")
        .optmulti(
            "",
            "skip",
            "skip tests matching SUBSTRING. Can be passed multiple times",
            "SUBSTRING",
        )
        .optflag("", "exact", "filters match exactly")
        .optopt(
            "",
            "runner",
            "supervisor program to run tests under \
             (eg. emulator, valgrind)",
            "PROGRAM",
        )
        .optmulti("", "host-rustcflags", "flags to pass to rustc for host", "FLAGS")
        .optmulti("", "target-rustcflags", "flags to pass to rustc for target", "FLAGS")
        .optflag(
            "",
            "rust-randomized-layout",
            "set this when rustc/stdlib were compiled with randomized layouts",
        )
        .optflag("", "optimize-tests", "run tests with optimizations enabled")
        .optflag("", "verbose", "run tests verbosely, showing all output")
        .optflag(
            "",
            "bless",
            "overwrite stderr/stdout files instead of complaining about a mismatch",
        )
        .optflag("", "quiet", "print one character per test instead of one line")
        .optopt("", "color", "coloring: auto, always, never", "WHEN")
        .optflag("", "json", "emit json output instead of plaintext output")
        .optopt("", "logfile", "file to log test execution to", "FILE")
        .optopt("", "target", "the target to build for", "TARGET")
        .optopt("", "host", "the host to build for", "HOST")
        .optopt("", "cdb", "path to CDB to use for CDB debuginfo tests", "PATH")
        .optopt("", "gdb", "path to GDB to use for GDB debuginfo tests", "PATH")
        .optopt("", "lldb-version", "the version of LLDB used", "VERSION STRING")
        .optopt("", "llvm-version", "the version of LLVM used", "VERSION STRING")
        .optflag("", "system-llvm", "is LLVM the system LLVM")
        .optopt("", "android-cross-path", "Android NDK standalone path", "PATH")
        .optopt("", "adb-path", "path to the android debugger", "PATH")
        .optopt("", "adb-test-dir", "path to tests for the android debugger", "PATH")
        .optopt("", "lldb-python-dir", "directory containing LLDB's python module", "PATH")
        .reqopt("", "cc", "path to a C compiler", "PATH")
        .reqopt("", "cxx", "path to a C++ compiler", "PATH")
        .reqopt("", "cflags", "flags for the C compiler", "FLAGS")
        .reqopt("", "cxxflags", "flags for the CXX compiler", "FLAGS")
        .optopt("", "ar", "path to an archiver", "PATH")
        .optopt("", "target-linker", "path to a linker for the target", "PATH")
        .optopt("", "host-linker", "path to a linker for the host", "PATH")
        .reqopt("", "llvm-components", "list of LLVM components built in", "LIST")
        .optopt("", "llvm-bin-dir", "Path to LLVM's `bin` directory", "PATH")
        .optopt("", "nodejs", "the name of nodejs", "PATH")
        .optopt("", "npm", "the name of npm", "PATH")
        .optopt("", "remote-test-client", "path to the remote test client", "PATH")
        .optopt(
            "",
            "compare-mode",
            "mode describing what file the actual ui output will be compared to",
            "COMPARE MODE",
        )
        .optflag(
            "",
            "rustfix-coverage",
            "enable this to generate a Rustfix coverage file, which is saved in \
            `./<build_base>/rustfix_missing_coverage.txt`",
        )
        .optflag("", "force-rerun", "rerun tests even if the inputs are unchanged")
        .optflag("", "only-modified", "only run tests that result been modified")
        .optflag("", "nocapture", "")
        .optflag("", "profiler-runtime", "is the profiler runtime enabled for this target")
        .optflag("h", "help", "show this message")
        .reqopt("", "channel", "current Rust channel", "CHANNEL")
        .optflag(
            "",
            "git-hash",
            "run tests which rely on commit version being compiled into the binaries",
        )
        .optopt("", "edition", "default Rust edition", "EDITION")
        .reqopt("", "git-repository", "name of the git repository", "ORG/REPO")
        .reqopt("", "nightly-branch", "name of the git branch for nightly", "BRANCH")
        .reqopt(
            "",
            "git-merge-commit-email",
            "email address used for finding merge commits",
            "EMAIL",
        )
        .optopt(
            "",
            "compiletest-diff-tool",
            "What custom diff tool to use for displaying compiletest tests.",
            "COMMAND",
        )
        .reqopt("", "minicore-path", "path to minicore aux library", "PATH");

    let (argv0, args_) = args.split_first().unwrap();
    if args.len() == 1 || args[1] == "-h" || args[1] == "--help" {
        let message = format!("Usage: {} [OPTIONS] [TESTNAME...]", argv0);
        println!("{}", opts.usage(&message));
        println!();
        panic!()
    }

    let matches = &match opts.parse(args_) {
        Ok(m) => m,
        Err(f) => panic!("{:?}", f),
    };

    if matches.opt_present("h") || matches.opt_present("help") {
        let message = format!("Usage: {} [OPTIONS]  [TESTNAME...]", argv0);
        println!("{}", opts.usage(&message));
        println!();
        panic!()
    }

    fn opt_path(m: &getopts::Matches, nm: &str) -> PathBuf {
        match m.opt_str(nm) {
            Some(s) => PathBuf::from(&s),
            None => panic!("no option (=path) found for {}", nm),
        }
    }

    fn make_absolute(path: PathBuf) -> PathBuf {
        if path.is_relative() { env::current_dir().unwrap().join(path) } else { path }
    }

    let target = opt_str2(matches.opt_str("target"));
    let android_cross_path = opt_path(matches, "android-cross-path");
    let (cdb, cdb_version) = debuggers::analyze_cdb(matches.opt_str("cdb"), &target);
    let (gdb, gdb_version) =
        debuggers::analyze_gdb(matches.opt_str("gdb"), &target, &android_cross_path);
    let lldb_version =
        matches.opt_str("lldb-version").as_deref().and_then(debuggers::extract_lldb_version);
    let color = match matches.opt_str("color").as_deref() {
        Some("auto") | None => ColorConfig::AutoColor,
        Some("always") => ColorConfig::AlwaysColor,
        Some("never") => ColorConfig::NeverColor,
        Some(x) => panic!("argument for --color must be auto, always, or never, but found `{}`", x),
    };
    let llvm_version =
        matches.opt_str("llvm-version").as_deref().map(header::extract_llvm_version).or_else(
            || header::extract_llvm_version_from_binary(&matches.opt_str("llvm-filecheck")?),
        );

    let src_base = opt_path(matches, "src-base");
    let run_ignored = matches.opt_present("ignored");
    let with_rustc_debug_assertions = matches.opt_present("with-rustc-debug-assertions");
    let with_std_debug_assertions = matches.opt_present("with-std-debug-assertions");
    let mode = matches.opt_str("mode").unwrap().parse().expect("invalid mode");
    let has_html_tidy = if mode == Mode::Rustdoc {
        Command::new("tidy")
            .arg("--version")
            .stdout(Stdio::null())
            .status()
            .map_or(false, |status| status.success())
    } else {
        // Avoid spawning an external command when we know html-tidy won't be used.
        false
    };
    let has_enzyme = matches.opt_present("has-enzyme");
    let filters = if mode == Mode::RunMake {
        matches
            .free
            .iter()
            .map(|f| {
                let path = Path::new(f);
                let mut iter = path.iter().skip(1);

                // We skip the test folder and check if the user passed `rmake.rs` or `Makefile`.
                if iter
                    .next()
                    .is_some_and(|s| s == OsStr::new("rmake.rs") || s == OsStr::new("Makefile"))
                    && iter.next().is_none()
                {
                    path.parent().unwrap().to_str().unwrap().to_string()
                } else {
                    f.to_string()
                }
            })
            .collect::<Vec<_>>()
    } else {
        matches.free.clone()
    };
    let compare_mode = matches.opt_str("compare-mode").map(|s| {
        s.parse().unwrap_or_else(|_| {
            let variants: Vec<_> = CompareMode::STR_VARIANTS.iter().copied().collect();
            panic!(
                "`{s}` is not a valid value for `--compare-mode`, it should be one of: {}",
                variants.join(", ")
            );
        })
    });
    Config {
        bless: matches.opt_present("bless"),
        compile_lib_path: make_absolute(opt_path(matches, "compile-lib-path")),
        run_lib_path: make_absolute(opt_path(matches, "run-lib-path")),
        rustc_path: opt_path(matches, "rustc-path"),
        cargo_path: matches.opt_str("cargo-path").map(PathBuf::from),
        rustdoc_path: matches.opt_str("rustdoc-path").map(PathBuf::from),
        coverage_dump_path: matches.opt_str("coverage-dump-path").map(PathBuf::from),
        python: matches.opt_str("python").unwrap(),
        jsondocck_path: matches.opt_str("jsondocck-path"),
        jsondoclint_path: matches.opt_str("jsondoclint-path"),
        run_clang_based_tests_with: matches.opt_str("run-clang-based-tests-with"),
        llvm_filecheck: matches.opt_str("llvm-filecheck").map(PathBuf::from),
        llvm_bin_dir: matches.opt_str("llvm-bin-dir").map(PathBuf::from),
        src_base,
        build_base: opt_path(matches, "build-base"),
        sysroot_base: opt_path(matches, "sysroot-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode,
        suite: matches.opt_str("suite").unwrap(),
        debugger: None,
        run_ignored,
        with_rustc_debug_assertions,
        with_std_debug_assertions,
        filters,
        skip: matches.opt_strs("skip"),
        filter_exact: matches.opt_present("exact"),
        force_pass_mode: matches.opt_str("pass").map(|mode| {
            mode.parse::<PassMode>()
                .unwrap_or_else(|_| panic!("unknown `--pass` option `{}` given", mode))
        }),
        run: matches.opt_str("run").and_then(|mode| match mode.as_str() {
            "auto" => None,
            "always" => Some(true),
            "never" => Some(false),
            _ => panic!("unknown `--run` option `{}` given", mode),
        }),
        logfile: matches.opt_str("logfile").map(|s| PathBuf::from(&s)),
        runner: matches.opt_str("runner"),
        host_rustcflags: matches.opt_strs("host-rustcflags"),
        target_rustcflags: matches.opt_strs("target-rustcflags"),
        optimize_tests: matches.opt_present("optimize-tests"),
        rust_randomized_layout: matches.opt_present("rust-randomized-layout"),
        target,
        host: opt_str2(matches.opt_str("host")),
        cdb,
        cdb_version,
        gdb,
        gdb_version,
        lldb_version,
        llvm_version,
        system_llvm: matches.opt_present("system-llvm"),
        android_cross_path,
        adb_path: opt_str2(matches.opt_str("adb-path")),
        adb_test_dir: opt_str2(matches.opt_str("adb-test-dir")),
        adb_device_status: opt_str2(matches.opt_str("target")).contains("android")
            && "(none)" != opt_str2(matches.opt_str("adb-test-dir"))
            && !opt_str2(matches.opt_str("adb-test-dir")).is_empty(),
        lldb_python_dir: matches.opt_str("lldb-python-dir"),
        verbose: matches.opt_present("verbose"),
        format: match (matches.opt_present("quiet"), matches.opt_present("json")) {
            (true, true) => panic!("--quiet and --json are incompatible"),
            (true, false) => test::OutputFormat::Terse,
            (false, true) => test::OutputFormat::Json,
            (false, false) => test::OutputFormat::Pretty,
        },
        only_modified: matches.opt_present("only-modified"),
        color,
        remote_test_client: matches.opt_str("remote-test-client").map(PathBuf::from),
        compare_mode,
        rustfix_coverage: matches.opt_present("rustfix-coverage"),
        has_html_tidy,
        has_enzyme,
        channel: matches.opt_str("channel").unwrap(),
        git_hash: matches.opt_present("git-hash"),
        edition: matches.opt_str("edition"),

        cc: matches.opt_str("cc").unwrap(),
        cxx: matches.opt_str("cxx").unwrap(),
        cflags: matches.opt_str("cflags").unwrap(),
        cxxflags: matches.opt_str("cxxflags").unwrap(),
        ar: matches.opt_str("ar").unwrap_or_else(|| String::from("ar")),
        target_linker: matches.opt_str("target-linker"),
        host_linker: matches.opt_str("host-linker"),
        llvm_components: matches.opt_str("llvm-components").unwrap(),
        nodejs: matches.opt_str("nodejs"),
        npm: matches.opt_str("npm"),

        force_rerun: matches.opt_present("force-rerun"),

        target_cfgs: OnceLock::new(),
        builtin_cfg_names: OnceLock::new(),

        nocapture: matches.opt_present("nocapture"),

        git_repository: matches.opt_str("git-repository").unwrap(),
        nightly_branch: matches.opt_str("nightly-branch").unwrap(),
        git_merge_commit_email: matches.opt_str("git-merge-commit-email").unwrap(),

        profiler_runtime: matches.opt_present("profiler-runtime"),

        diff_command: matches.opt_str("compiletest-diff-tool"),

        minicore_path: opt_path(matches, "minicore-path"),
    }
}

pub fn log_config(config: &Config) {
    let c = config;
    logv(c, "configuration:".to_string());
    logv(c, format!("compile_lib_path: {:?}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {:?}", config.run_lib_path));
    logv(c, format!("rustc_path: {:?}", config.rustc_path.display()));
    logv(c, format!("cargo_path: {:?}", config.cargo_path));
    logv(c, format!("rustdoc_path: {:?}", config.rustdoc_path));
    logv(c, format!("src_base: {:?}", config.src_base.display()));
    logv(c, format!("build_base: {:?}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", config.mode));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filters: {:?}", config.filters));
    logv(c, format!("skip: {:?}", config.skip));
    logv(c, format!("filter_exact: {}", config.filter_exact));
    logv(
        c,
        format!("force_pass_mode: {}", opt_str(&config.force_pass_mode.map(|m| format!("{}", m))),),
    );
    logv(c, format!("runner: {}", opt_str(&config.runner)));
    logv(c, format!("host-rustcflags: {:?}", config.host_rustcflags));
    logv(c, format!("target-rustcflags: {:?}", config.target_rustcflags));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("android-cross-path: {:?}", config.android_cross_path.display()));
    logv(c, format!("adb_path: {:?}", config.adb_path));
    logv(c, format!("adb_test_dir: {:?}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}", config.adb_device_status));
    logv(c, format!("ar: {}", config.ar));
    logv(c, format!("target-linker: {:?}", config.target_linker));
    logv(c, format!("host-linker: {:?}", config.host_linker));
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("format: {:?}", config.format));
    logv(c, format!("minicore_path: {:?}", config.minicore_path.display()));
    logv(c, "\n".to_string());
}

pub fn opt_str(maybestr: &Option<String>) -> &str {
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

/// Called by `main` after the config has been parsed.
pub fn run_tests(config: Arc<Config>) {
    // If we want to collect rustfix coverage information,
    // we first make sure that the coverage file does not exist.
    // It will be created later on.
    if config.rustfix_coverage {
        let mut coverage_file_path = config.build_base.clone();
        coverage_file_path.push("rustfix_missing_coverage.txt");
        if coverage_file_path.exists() {
            if let Err(e) = fs::remove_file(&coverage_file_path) {
                panic!("Could not delete {} due to {}", coverage_file_path.display(), e)
            }
        }
    }

    // sadly osx needs some file descriptor limits raised for running tests in
    // parallel (especially when we have lots and lots of child processes).
    // For context, see #8904
    unsafe {
        raise_fd_limit::raise_fd_limit();
    }
    // Prevent issue #21352 UAC blocking .exe containing 'patch' etc. on Windows
    // If #11207 is resolved (adding manifest to .exe) this becomes unnecessary
    env::set_var("__COMPAT_LAYER", "RunAsInvoker");

    // Let tests know which target they're running as
    env::set_var("TARGET", &config.target);

    let opts = test_opts(&config);

    let mut configs = Vec::new();
    if let Mode::DebugInfo = config.mode {
        // Debugging emscripten code doesn't make sense today
        if !config.target.contains("emscripten") {
            configs.extend(debuggers::configure_cdb(&config));
            configs.extend(debuggers::configure_gdb(&config));
            configs.extend(debuggers::configure_lldb(&config));
        }
    } else {
        configs.push(config.clone());
    };

    // Discover all of the tests in the test suite directory, and build a libtest
    // structure for each test (or each revision of a multi-revision test).
    let mut tests = Vec::new();
    for c in configs {
        tests.extend(collect_and_make_tests(c));
    }

    tests.sort_by(|a, b| a.desc.name.as_slice().cmp(&b.desc.name.as_slice()));

    // Delegate to libtest to filter and run the big list of structures created
    // during test discovery. When libtest decides to run a test, it will invoke
    // the corresponding closure created by `make_test_closure`.
    let res = test::run_tests_console(&opts, tests);

    // Check the outcome reported by libtest.
    match res {
        Ok(true) => {}
        Ok(false) => {
            // We want to report that the tests failed, but we also want to give
            // some indication of just what tests we were running. Especially on
            // CI, where there can be cross-compiled tests for a lot of
            // architectures, without this critical information it can be quite
            // easy to miss which tests failed, and as such fail to reproduce
            // the failure locally.

            println!(
                "Some tests failed in compiletest suite={}{} mode={} host={} target={}",
                config.suite,
                config
                    .compare_mode
                    .as_ref()
                    .map(|c| format!(" compare_mode={:?}", c))
                    .unwrap_or_default(),
                config.mode,
                config.host,
                config.target
            );

            std::process::exit(1);
        }
        Err(e) => {
            // We don't know if tests passed or not, but if there was an error
            // during testing we don't want to just succeed (we may not have
            // tested something), so fail.
            //
            // This should realistically "never" happen, so don't try to make
            // this a pretty error message.
            panic!("I/O failure during tests: {:?}", e);
        }
    }
}

pub fn test_opts(config: &Config) -> test::TestOpts {
    if env::var("RUST_TEST_NOCAPTURE").is_ok() {
        eprintln!(
            "WARNING: RUST_TEST_NOCAPTURE is no longer used. \
                   Use the `--nocapture` flag instead."
        );
    }

    test::TestOpts {
        exclude_should_panic: false,
        filters: config.filters.clone(),
        filter_exact: config.filter_exact,
        run_ignored: if config.run_ignored { test::RunIgnored::Yes } else { test::RunIgnored::No },
        format: config.format,
        logfile: config.logfile.clone(),
        run_tests: true,
        bench_benchmarks: true,
        nocapture: config.nocapture,
        color: config.color,
        shuffle: false,
        shuffle_seed: None,
        test_threads: None,
        skip: config.skip.clone(),
        list: false,
        options: test::Options::new(),
        time_options: None,
        force_run_in_process: false,
        fail_fast: std::env::var_os("RUSTC_TEST_FAIL_FAST").is_some(),
    }
}

/// Read-only context data used during test collection.
struct TestCollectorCx {
    config: Arc<Config>,
    cache: HeadersCache,
    common_inputs_stamp: Stamp,
    modified_tests: Vec<PathBuf>,
}

/// Mutable state used during test collection.
struct TestCollector {
    tests: Vec<test::TestDescAndFn>,
    found_path_stems: HashSet<PathBuf>,
    poisoned: bool,
}

/// Creates libtest structures for every test/revision in the test suite directory.
///
/// This always inspects _all_ test files in the suite (e.g. all 17k+ ui tests),
/// regardless of whether any filters/tests were specified on the command-line,
/// because filtering is handled later by libtest.
pub fn collect_and_make_tests(config: Arc<Config>) -> Vec<test::TestDescAndFn> {
    debug!("making tests from {:?}", config.src_base.display());
    let common_inputs_stamp = common_inputs_stamp(&config);
    let modified_tests = modified_tests(&config, &config.src_base).unwrap_or_else(|err| {
        panic!("modified_tests got error from dir: {}, error: {}", config.src_base.display(), err)
    });
    let cache = HeadersCache::load(&config);

    let cx = TestCollectorCx { config, cache, common_inputs_stamp, modified_tests };
    let mut collector =
        TestCollector { tests: vec![], found_path_stems: HashSet::new(), poisoned: false };

    collect_tests_from_dir(&cx, &mut collector, &cx.config.src_base, Path::new("")).unwrap_or_else(
        |reason| panic!("Could not read tests from {}: {reason}", cx.config.src_base.display()),
    );

    let TestCollector { tests, found_path_stems, poisoned } = collector;

    if poisoned {
        eprintln!();
        panic!("there are errors in tests");
    }

    check_for_overlapping_test_paths(&found_path_stems);

    tests
}

/// Returns the most recent last-modified timestamp from among the input files
/// that are considered relevant to all tests (e.g. the compiler, std, and
/// compiletest itself).
///
/// (Some of these inputs aren't actually relevant to _all_ tests, but they are
/// common to some subset of tests, and are hopefully unlikely to be modified
/// while working on other tests.)
fn common_inputs_stamp(config: &Config) -> Stamp {
    let rust_src_dir = config.find_rust_src_root().expect("Could not find Rust source root");

    let mut stamp = Stamp::from_path(&config.rustc_path);

    // Relevant pretty printer files
    let pretty_printer_files = [
        "src/etc/rust_types.py",
        "src/etc/gdb_load_rust_pretty_printers.py",
        "src/etc/gdb_lookup.py",
        "src/etc/gdb_providers.py",
        "src/etc/lldb_batchmode.py",
        "src/etc/lldb_lookup.py",
        "src/etc/lldb_providers.py",
    ];
    for file in &pretty_printer_files {
        let path = rust_src_dir.join(file);
        stamp.add_path(&path);
    }

    stamp.add_dir(&rust_src_dir.join("src/etc/natvis"));

    stamp.add_dir(&config.run_lib_path);

    if let Some(ref rustdoc_path) = config.rustdoc_path {
        stamp.add_path(&rustdoc_path);
        stamp.add_path(&rust_src_dir.join("src/etc/htmldocck.py"));
    }

    // Re-run coverage tests if the `coverage-dump` tool was modified,
    // because its output format might have changed.
    if let Some(coverage_dump_path) = &config.coverage_dump_path {
        stamp.add_path(coverage_dump_path)
    }

    stamp.add_dir(&rust_src_dir.join("src/tools/run-make-support"));

    // Compiletest itself.
    stamp.add_dir(&rust_src_dir.join("src/tools/compiletest/"));

    stamp
}

/// Returns a list of modified/untracked test files that should be run when
/// the `--only-modified` flag is in use.
///
/// (Might be inaccurate in some cases.)
fn modified_tests(config: &Config, dir: &Path) -> Result<Vec<PathBuf>, String> {
    // If `--only-modified` wasn't passed, the list of modified tests won't be
    // used for anything, so avoid some work and just return an empty list.
    if !config.only_modified {
        return Ok(vec![]);
    }

    let files =
        get_git_modified_files(&config.git_config(), Some(dir), &vec!["rs", "stderr", "fixed"])?
            .unwrap_or(vec![]);
    // Add new test cases to the list, it will be convenient in daily development.
    let untracked_files = get_git_untracked_files(&config.git_config(), None)?.unwrap_or(vec![]);

    let all_paths = [&files[..], &untracked_files[..]].concat();
    let full_paths = {
        let mut full_paths: Vec<PathBuf> = all_paths
            .into_iter()
            .map(|f| PathBuf::from(f).with_extension("").with_extension("rs"))
            .filter_map(|f| if Path::new(&f).exists() { f.canonicalize().ok() } else { None })
            .collect();
        full_paths.dedup();
        full_paths.sort_unstable();
        full_paths
    };
    Ok(full_paths)
}

/// Recursively scans a directory to find test files and create test structures
/// that will be handed over to libtest.
fn collect_tests_from_dir(
    cx: &TestCollectorCx,
    collector: &mut TestCollector,
    dir: &Path,
    relative_dir_path: &Path,
) -> io::Result<()> {
    // Ignore directories that contain a file named `compiletest-ignore-dir`.
    if dir.join("compiletest-ignore-dir").exists() {
        return Ok(());
    }

    // For run-make tests, a "test file" is actually a directory that contains
    // an `rmake.rs` or `Makefile`"
    if cx.config.mode == Mode::RunMake {
        if dir.join("Makefile").exists() && dir.join("rmake.rs").exists() {
            return Err(io::Error::other(
                "run-make tests cannot have both `Makefile` and `rmake.rs`",
            ));
        }

        if dir.join("Makefile").exists() || dir.join("rmake.rs").exists() {
            let paths = TestPaths {
                file: dir.to_path_buf(),
                relative_dir: relative_dir_path.parent().unwrap().to_path_buf(),
            };
            make_test(cx, collector, &paths);
            // This directory is a test, so don't try to find other tests inside it.
            return Ok(());
        }
    }

    // If we find a test foo/bar.rs, we have to build the
    // output directory `$build/foo` so we can write
    // `$build/foo/bar` into it. We do this *now* in this
    // sequential loop because otherwise, if we do it in the
    // tests themselves, they race for the privilege of
    // creating the directories and sometimes fail randomly.
    let build_dir = output_relative_path(&cx.config, relative_dir_path);
    fs::create_dir_all(&build_dir).unwrap();

    // Add each `.rs` file as a test, and recurse further on any
    // subdirectories we find, except for `auxiliary` directories.
    // FIXME: this walks full tests tree, even if we have something to ignore
    // use walkdir/ignore like in tidy?
    for file in fs::read_dir(dir)? {
        let file = file?;
        let file_path = file.path();
        let file_name = file.file_name();

        if is_test(&file_name)
            && (!cx.config.only_modified || cx.modified_tests.contains(&file_path))
        {
            // We found a test file, so create the corresponding libtest structures.
            debug!("found test file: {:?}", file_path.display());

            // Record the stem of the test file, to check for overlaps later.
            let rel_test_path = relative_dir_path.join(file_path.file_stem().unwrap());
            collector.found_path_stems.insert(rel_test_path);

            let paths =
                TestPaths { file: file_path, relative_dir: relative_dir_path.to_path_buf() };
            make_test(cx, collector, &paths);
        } else if file_path.is_dir() {
            // Recurse to find more tests in a subdirectory.
            let relative_file_path = relative_dir_path.join(file.file_name());
            if &file_name != "auxiliary" {
                debug!("found directory: {:?}", file_path.display());
                collect_tests_from_dir(cx, collector, &file_path, &relative_file_path)?;
            }
        } else {
            debug!("found other file/directory: {:?}", file_path.display());
        }
    }
    Ok(())
}

/// Returns true if `file_name` looks like a proper test file name.
pub fn is_test(file_name: &OsString) -> bool {
    let file_name = file_name.to_str().unwrap();

    if !file_name.ends_with(".rs") {
        return false;
    }

    // `.`, `#`, and `~` are common temp-file prefixes.
    let invalid_prefixes = &[".", "#", "~"];
    !invalid_prefixes.iter().any(|p| file_name.starts_with(p))
}

/// For a single test file, creates one or more test structures (one per revision)
/// that can be handed over to libtest to run, possibly in parallel.
fn make_test(cx: &TestCollectorCx, collector: &mut TestCollector, testpaths: &TestPaths) {
    // For run-make tests, each "test file" is actually a _directory_ containing
    // an `rmake.rs` or `Makefile`. But for the purposes of directive parsing,
    // we want to look at that recipe file, not the directory itself.
    let test_path = if cx.config.mode == Mode::RunMake {
        if testpaths.file.join("rmake.rs").exists() && testpaths.file.join("Makefile").exists() {
            panic!("run-make tests cannot have both `rmake.rs` and `Makefile`");
        }

        if testpaths.file.join("rmake.rs").exists() {
            // Parse directives in rmake.rs.
            testpaths.file.join("rmake.rs")
        } else {
            // Parse directives in the Makefile.
            testpaths.file.join("Makefile")
        }
    } else {
        PathBuf::from(&testpaths.file)
    };

    // Scan the test file to discover its revisions, if any.
    let early_props = EarlyProps::from_file(&cx.config, &test_path);

    // Normally we create one libtest structure per revision, with two exceptions:
    // - If a test doesn't use revisions, create a dummy revision (None) so that
    //   the test can still run.
    // - Incremental tests inherently can't run their revisions in parallel, so
    //   we treat them like non-revisioned tests here. Incremental revisions are
    //   handled internally by `runtest::run` instead.
    let revisions = if early_props.revisions.is_empty() || cx.config.mode == Mode::Incremental {
        vec![None]
    } else {
        early_props.revisions.iter().map(|r| Some(r.as_str())).collect()
    };

    // For each revision (or the sole dummy revision), create and append a
    // `test::TestDescAndFn` that can be handed over to libtest.
    collector.tests.extend(revisions.into_iter().map(|revision| {
        // Create a test name and description to hand over to libtest.
        let src_file = fs::File::open(&test_path).expect("open test file to parse ignores");
        let test_name = make_test_name(&cx.config, testpaths, revision);
        // Create a libtest description for the test/revision.
        // This is where `ignore-*`/`only-*`/`needs-*` directives are handled,
        // because they need to set the libtest ignored flag.
        let mut desc = make_test_description(
            &cx.config,
            &cx.cache,
            test_name,
            &test_path,
            src_file,
            revision,
            &mut collector.poisoned,
        );

        // If a test's inputs haven't changed since the last time it ran,
        // mark it as ignored so that libtest will skip it.
        if !cx.config.force_rerun && is_up_to_date(cx, testpaths, &early_props, revision) {
            desc.ignore = true;
            // Keep this in sync with the "up-to-date" message detected by bootstrap.
            desc.ignore_message = Some("up-to-date");
        }

        // Create the callback that will run this test/revision when libtest calls it.
        let testfn = make_test_closure(Arc::clone(&cx.config), testpaths, revision);

        test::TestDescAndFn { desc, testfn }
    }));
}

/// The path of the `stamp` file that gets created or updated whenever a
/// particular test completes successfully.
fn stamp_file_path(config: &Config, testpaths: &TestPaths, revision: Option<&str>) -> PathBuf {
    output_base_dir(config, testpaths, revision).join("stamp")
}

/// Returns a list of files that, if modified, would cause this test to no
/// longer be up-to-date.
///
/// (Might be inaccurate in some cases.)
fn files_related_to_test(
    config: &Config,
    testpaths: &TestPaths,
    props: &EarlyProps,
    revision: Option<&str>,
) -> Vec<PathBuf> {
    let mut related = vec![];

    if testpaths.file.is_dir() {
        // run-make tests use their individual directory
        for entry in WalkDir::new(&testpaths.file) {
            let path = entry.unwrap().into_path();
            if path.is_file() {
                related.push(path);
            }
        }
    } else {
        related.push(testpaths.file.clone());
    }

    for aux in props.aux.all_aux_path_strings() {
        // FIXME(Zalathar): Perform all `auxiliary` path resolution in one place.
        let path = testpaths.file.parent().unwrap().join("auxiliary").join(aux);
        related.push(path);
    }

    // UI test files.
    for extension in UI_EXTENSIONS {
        let path = expected_output_path(testpaths, revision, &config.compare_mode, extension);
        related.push(path);
    }

    // `minicore.rs` test auxiliary: we need to make sure tests get rerun if this changes.
    //
    // FIXME(jieyouxu): untangle these paths, we should provide both a path to root `tests/` or
    // `tests/auxiliary/` and the test suite in question. `src_base` is also a terrible name.
    related.push(config.src_base.parent().unwrap().join("auxiliary").join("minicore.rs"));

    related
}

/// Checks whether a particular test/revision is "up-to-date", meaning that no
/// relevant files/settings have changed since the last time the test succeeded.
///
/// (This is not very reliable in some circumstances, so the `--force-rerun`
/// flag can be used to ignore up-to-date checking and always re-run tests.)
fn is_up_to_date(
    cx: &TestCollectorCx,
    testpaths: &TestPaths,
    props: &EarlyProps,
    revision: Option<&str>,
) -> bool {
    let stamp_file_path = stamp_file_path(&cx.config, testpaths, revision);
    // Check the config hash inside the stamp file.
    let contents = match fs::read_to_string(&stamp_file_path) {
        Ok(f) => f,
        Err(ref e) if e.kind() == ErrorKind::InvalidData => panic!("Can't read stamp contents"),
        // The test hasn't succeeded yet, so it is not up-to-date.
        Err(_) => return false,
    };
    let expected_hash = runtest::compute_stamp_hash(&cx.config);
    if contents != expected_hash {
        // Some part of compiletest configuration has changed since the test
        // last succeeded, so it is not up-to-date.
        return false;
    }

    // Check the timestamp of the stamp file against the last modified time
    // of all files known to be relevant to the test.
    let mut inputs_stamp = cx.common_inputs_stamp.clone();
    for path in files_related_to_test(&cx.config, testpaths, props, revision) {
        inputs_stamp.add_path(&path);
    }

    // If no relevant files have been modified since the stamp file was last
    // written, the test is up-to-date.
    inputs_stamp < Stamp::from_path(&stamp_file_path)
}

/// The maximum of a set of file-modified timestamps.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Stamp {
    time: SystemTime,
}

impl Stamp {
    /// Creates a timestamp holding the last-modified time of the specified file.
    fn from_path(path: &Path) -> Self {
        let mut stamp = Stamp { time: SystemTime::UNIX_EPOCH };
        stamp.add_path(path);
        stamp
    }

    /// Updates this timestamp to the last-modified time of the specified file,
    /// if it is later than the currently-stored timestamp.
    fn add_path(&mut self, path: &Path) {
        let modified = fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.time = self.time.max(modified);
    }

    /// Updates this timestamp to the most recent last-modified time of all files
    /// recursively contained in the given directory, if it is later than the
    /// currently-stored timestamp.
    fn add_dir(&mut self, path: &Path) {
        for entry in WalkDir::new(path) {
            let entry = entry.unwrap();
            if entry.file_type().is_file() {
                let modified = entry
                    .metadata()
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                self.time = self.time.max(modified);
            }
        }
    }
}

/// Creates a name for this test/revision that can be handed over to libtest.
fn make_test_name(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> test::TestName {
    // Print the name of the file, relative to the repository root.
    // `src_base` looks like `/path/to/rust/tests/ui`
    let root_directory = config.src_base.parent().unwrap().parent().unwrap();
    let path = testpaths.file.strip_prefix(root_directory).unwrap();
    let debugger = match config.debugger {
        Some(d) => format!("-{}", d),
        None => String::new(),
    };
    let mode_suffix = match config.compare_mode {
        Some(ref mode) => format!(" ({})", mode.to_str()),
        None => String::new(),
    };

    test::DynTestName(format!(
        "[{}{}{}] {}{}",
        config.mode,
        debugger,
        mode_suffix,
        path.display(),
        revision.map_or("".to_string(), |rev| format!("#{}", rev))
    ))
}

/// Creates a callback for this test/revision that libtest will call when it
/// decides to actually run the underlying test.
fn make_test_closure(
    config: Arc<Config>,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> test::TestFn {
    let testpaths = testpaths.clone();
    let revision = revision.map(str::to_owned);

    // This callback is the link between compiletest's test discovery code,
    // and the parts of compiletest that know how to run an individual test.
    test::DynTestFn(Box::new(move || {
        runtest::run(config, &testpaths, revision.as_deref());
        Ok(())
    }))
}

/// Checks that test discovery didn't find any tests whose name stem is a prefix
/// of some other tests's name.
///
/// For example, suppose the test suite contains these two test files:
/// - `tests/rustdoc/primitive.rs`
/// - `tests/rustdoc/primitive/no_std.rs`
///
/// The test runner might put the output from those tests in these directories:
/// - `$build/test/rustdoc/primitive/`
/// - `$build/test/rustdoc/primitive/no_std/`
///
/// Because one output path is a subdirectory of the other, the two tests might
/// interfere with each other in unwanted ways, especially if the test runner
/// decides to delete test output directories to clean them between runs.
/// To avoid problems, we forbid test names from overlapping in this way.
///
/// See <https://github.com/rust-lang/rust/pull/109509> for more context.
fn check_for_overlapping_test_paths(found_path_stems: &HashSet<PathBuf>) {
    let mut collisions = Vec::new();
    for path in found_path_stems {
        for ancestor in path.ancestors().skip(1) {
            if found_path_stems.contains(ancestor) {
                collisions.push((path, ancestor));
            }
        }
    }
    if !collisions.is_empty() {
        collisions.sort();
        let collisions: String = collisions
            .into_iter()
            .map(|(path, check_parent)| format!("test {path:?} clashes with {check_parent:?}\n"))
            .collect();
        panic!(
            "{collisions}\n\
            Tests cannot have overlapping names. Make sure they use unique prefixes."
        );
    }
}
