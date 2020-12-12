#![crate_name = "compiletest"]
// The `test` crate is the only unstable feature
// allowed here, just to share similar code.
#![feature(test)]

extern crate test;

use crate::common::{expected_output_path, output_base_dir, output_relative_path, UI_EXTENSIONS};
use crate::common::{CompareMode, Config, Debugger, Mode, PassMode, Pretty, TestPaths};
use crate::util::logv;
use getopts::Options;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;
use test::ColorConfig;
use tracing::*;
use walkdir::WalkDir;

use self::header::EarlyProps;

#[cfg(test)]
mod tests;

pub mod common;
pub mod errors;
pub mod header;
mod json;
mod raise_fd_limit;
mod read2;
pub mod runtest;
pub mod util;

fn main() {
    tracing_subscriber::fmt::init();

    let config = parse_config(env::args().collect());

    if config.valgrind_path.is_none() && config.force_valgrind {
        panic!("Can't find Valgrind to run Valgrind tests");
    }

    if !config.has_tidy && config.mode == Mode::Rustdoc {
        eprintln!("warning: `tidy` is not installed; generated diffs will be harder to read");
    }

    log_config(&config);
    run_tests(config);
}

pub fn parse_config(args: Vec<String>) -> Config {
    let mut opts = Options::new();
    opts.reqopt("", "compile-lib-path", "path to host shared libraries", "PATH")
        .reqopt("", "run-lib-path", "path to target shared libraries", "PATH")
        .reqopt("", "rustc-path", "path to rustc to use for compiling", "PATH")
        .optopt("", "rustdoc-path", "path to rustdoc to use for compiling", "PATH")
        .optopt("", "rust-demangler-path", "path to rust-demangler to use in tests", "PATH")
        .reqopt("", "lldb-python", "path to python to use for doc tests", "PATH")
        .reqopt("", "docck-python", "path to python to use for doc tests", "PATH")
        .optopt("", "valgrind-path", "path to Valgrind executable for Valgrind tests", "PROGRAM")
        .optflag("", "force-valgrind", "fail if Valgrind tests cannot be run under Valgrind")
        .optopt("", "run-clang-based-tests-with", "path to Clang executable", "PATH")
        .optopt("", "llvm-filecheck", "path to LLVM's FileCheck binary", "DIR")
        .reqopt("", "src-base", "directory to scan for test files", "PATH")
        .reqopt("", "build-base", "directory to deposit test outputs", "PATH")
        .reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET")
        .reqopt(
            "",
            "mode",
            "which sort of compile tests to run",
            "compile-fail | run-fail | run-pass-valgrind | pretty | debug-info | codegen | rustdoc \
            | rustdoc-json | codegen-units | incremental | run-make | ui | js-doc-test | mir-opt | assembly",
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
        .optflag("", "ignored", "run tests marked as ignored")
        .optflag("", "exact", "filters match exactly")
        .optopt(
            "",
            "runtool",
            "supervisor program to run tests under \
             (eg. emulator, valgrind)",
            "PROGRAM",
        )
        .optopt("", "host-rustcflags", "flags to pass to rustc for host", "FLAGS")
        .optopt("", "target-rustcflags", "flags to pass to rustc for target", "FLAGS")
        .optflag("", "verbose", "run tests verbosely, showing all output")
        .optflag(
            "",
            "bless",
            "overwrite stderr/stdout files instead of complaining about a mismatch",
        )
        .optflag("", "quiet", "print one character per test instead of one line")
        .optopt("", "color", "coloring: auto, always, never", "WHEN")
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
        .optopt("", "ar", "path to an archiver", "PATH")
        .optopt("", "linker", "path to a linker", "PATH")
        .reqopt("", "llvm-components", "list of LLVM components built in", "LIST")
        .optopt("", "llvm-bin-dir", "Path to LLVM's `bin` directory", "PATH")
        .optopt("", "nodejs", "the name of nodejs", "PATH")
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
        .optflag("h", "help", "show this message");

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
    let (cdb, cdb_version) = analyze_cdb(matches.opt_str("cdb"), &target);
    let (gdb, gdb_version, gdb_native_rust) =
        analyze_gdb(matches.opt_str("gdb"), &target, &android_cross_path);
    let (lldb_version, lldb_native_rust) = matches
        .opt_str("lldb-version")
        .as_deref()
        .and_then(extract_lldb_version)
        .map(|(v, b)| (Some(v), b))
        .unwrap_or((None, false));
    let color = match matches.opt_str("color").as_deref() {
        Some("auto") | None => ColorConfig::AutoColor,
        Some("always") => ColorConfig::AlwaysColor,
        Some("never") => ColorConfig::NeverColor,
        Some(x) => panic!("argument for --color must be auto, always, or never, but found `{}`", x),
    };
    let llvm_version =
        matches.opt_str("llvm-version").as_deref().and_then(header::extract_llvm_version);

    let src_base = opt_path(matches, "src-base");
    let run_ignored = matches.opt_present("ignored");
    let has_tidy = Command::new("tidy")
        .arg("--version")
        .stdout(Stdio::null())
        .status()
        .map_or(false, |status| status.success());
    Config {
        bless: matches.opt_present("bless"),
        compile_lib_path: make_absolute(opt_path(matches, "compile-lib-path")),
        run_lib_path: make_absolute(opt_path(matches, "run-lib-path")),
        rustc_path: opt_path(matches, "rustc-path"),
        rustdoc_path: matches.opt_str("rustdoc-path").map(PathBuf::from),
        rust_demangler_path: matches.opt_str("rust-demangler-path").map(PathBuf::from),
        lldb_python: matches.opt_str("lldb-python").unwrap(),
        docck_python: matches.opt_str("docck-python").unwrap(),
        valgrind_path: matches.opt_str("valgrind-path"),
        force_valgrind: matches.opt_present("force-valgrind"),
        run_clang_based_tests_with: matches.opt_str("run-clang-based-tests-with"),
        llvm_filecheck: matches.opt_str("llvm-filecheck").map(PathBuf::from),
        llvm_bin_dir: matches.opt_str("llvm-bin-dir").map(PathBuf::from),
        src_base,
        build_base: opt_path(matches, "build-base"),
        stage_id: matches.opt_str("stage-id").unwrap(),
        mode: matches.opt_str("mode").unwrap().parse().expect("invalid mode"),
        suite: matches.opt_str("suite").unwrap(),
        debugger: None,
        run_ignored,
        filter: matches.free.first().cloned(),
        filter_exact: matches.opt_present("exact"),
        force_pass_mode: matches.opt_str("pass").map(|mode| {
            mode.parse::<PassMode>()
                .unwrap_or_else(|_| panic!("unknown `--pass` option `{}` given", mode))
        }),
        logfile: matches.opt_str("logfile").map(|s| PathBuf::from(&s)),
        runtool: matches.opt_str("runtool"),
        host_rustcflags: matches.opt_str("host-rustcflags"),
        target_rustcflags: matches.opt_str("target-rustcflags"),
        target,
        host: opt_str2(matches.opt_str("host")),
        cdb,
        cdb_version,
        gdb,
        gdb_version,
        gdb_native_rust,
        lldb_version,
        lldb_native_rust,
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
        quiet: matches.opt_present("quiet"),
        color,
        remote_test_client: matches.opt_str("remote-test-client").map(PathBuf::from),
        compare_mode: matches.opt_str("compare-mode").map(CompareMode::parse),
        rustfix_coverage: matches.opt_present("rustfix-coverage"),
        has_tidy,

        cc: matches.opt_str("cc").unwrap(),
        cxx: matches.opt_str("cxx").unwrap(),
        cflags: matches.opt_str("cflags").unwrap(),
        ar: matches.opt_str("ar").unwrap_or_else(|| String::from("ar")),
        linker: matches.opt_str("linker"),
        llvm_components: matches.opt_str("llvm-components").unwrap(),
        nodejs: matches.opt_str("nodejs"),
    }
}

pub fn log_config(config: &Config) {
    let c = config;
    logv(c, "configuration:".to_string());
    logv(c, format!("compile_lib_path: {:?}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {:?}", config.run_lib_path));
    logv(c, format!("rustc_path: {:?}", config.rustc_path.display()));
    logv(c, format!("rustdoc_path: {:?}", config.rustdoc_path));
    logv(c, format!("rust_demangler_path: {:?}", config.rust_demangler_path));
    logv(c, format!("src_base: {:?}", config.src_base.display()));
    logv(c, format!("build_base: {:?}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", config.mode));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filter: {}", opt_str(&config.filter)));
    logv(c, format!("filter_exact: {}", config.filter_exact));
    logv(
        c,
        format!("force_pass_mode: {}", opt_str(&config.force_pass_mode.map(|m| format!("{}", m))),),
    );
    logv(c, format!("runtool: {}", opt_str(&config.runtool)));
    logv(c, format!("host-rustcflags: {}", opt_str(&config.host_rustcflags)));
    logv(c, format!("target-rustcflags: {}", opt_str(&config.target_rustcflags)));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("android-cross-path: {:?}", config.android_cross_path.display()));
    logv(c, format!("adb_path: {:?}", config.adb_path));
    logv(c, format!("adb_test_dir: {:?}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}", config.adb_device_status));
    logv(c, format!("ar: {}", config.ar));
    logv(c, format!("linker: {:?}", config.linker));
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("quiet: {}", config.quiet));
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

pub fn run_tests(config: Config) {
    // FIXME(#33435) Avoid spurious failures in codegen-units/partitioning tests.
    if let Mode::CodegenUnits = config.mode {
        let _ = fs::remove_dir_all("tmp/partitioning-tests");
    }

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
            configs.extend(configure_cdb(&config));
            configs.extend(configure_gdb(&config));
            configs.extend(configure_lldb(&config));
        }
    } else {
        configs.push(config.clone());
    };

    let mut tests = Vec::new();
    for c in &configs {
        make_tests(c, &mut tests);
    }

    let res = test::run_tests_console(&opts, tests);
    match res {
        Ok(true) => {}
        Ok(false) => {
            // We want to report that the tests failed, but we also want to give
            // some indication of just what tests we were running. Especially on
            // CI, where there can be cross-compiled tests for a lot of
            // architectures, without this critical information it can be quite
            // easy to miss which tests failed, and as such fail to reproduce
            // the failure locally.

            eprintln!(
                "Some tests failed in compiletest suite={}{} mode={} host={} target={}",
                config.suite,
                config.compare_mode.map(|c| format!(" compare_mode={:?}", c)).unwrap_or_default(),
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

fn configure_cdb(config: &Config) -> Option<Config> {
    config.cdb.as_ref()?;

    Some(Config { debugger: Some(Debugger::Cdb), ..config.clone() })
}

fn configure_gdb(config: &Config) -> Option<Config> {
    config.gdb_version?;

    if util::matches_env(&config.target, "msvc") {
        return None;
    }

    if config.remote_test_client.is_some() && !config.target.contains("android") {
        println!(
            "WARNING: debuginfo tests are not available when \
             testing with remote"
        );
        return None;
    }

    if config.target.contains("android") {
        println!(
            "{} debug-info test uses tcp 5039 port.\
             please reserve it",
            config.target
        );

        // android debug-info test uses remote debugger so, we test 1 thread
        // at once as they're all sharing the same TCP port to communicate
        // over.
        //
        // we should figure out how to lift this restriction! (run them all
        // on different ports allocated dynamically).
        env::set_var("RUST_TEST_THREADS", "1");
    }

    Some(Config { debugger: Some(Debugger::Gdb), ..config.clone() })
}

fn configure_lldb(config: &Config) -> Option<Config> {
    config.lldb_python_dir.as_ref()?;

    if let Some(350) = config.lldb_version {
        println!(
            "WARNING: The used version of LLDB (350) has a \
             known issue that breaks debuginfo tests. See \
             issue #32520 for more information. Skipping all \
             LLDB-based tests!",
        );
        return None;
    }

    // Some older versions of LLDB seem to have problems with multiple
    // instances running in parallel, so only run one test thread at a
    // time.
    env::set_var("RUST_TEST_THREADS", "1");

    Some(Config { debugger: Some(Debugger::Lldb), ..config.clone() })
}

pub fn test_opts(config: &Config) -> test::TestOpts {
    test::TestOpts {
        exclude_should_panic: false,
        filter: config.filter.clone(),
        filter_exact: config.filter_exact,
        run_ignored: if config.run_ignored { test::RunIgnored::Yes } else { test::RunIgnored::No },
        format: if config.quiet { test::OutputFormat::Terse } else { test::OutputFormat::Pretty },
        logfile: config.logfile.clone(),
        run_tests: true,
        bench_benchmarks: true,
        nocapture: match env::var("RUST_TEST_NOCAPTURE") {
            Ok(val) => &val != "0",
            Err(_) => false,
        },
        color: config.color,
        test_threads: None,
        skip: vec![],
        list: false,
        options: test::Options::new(),
        time_options: None,
        force_run_in_process: false,
    }
}

pub fn make_tests(config: &Config, tests: &mut Vec<test::TestDescAndFn>) {
    debug!("making tests from {:?}", config.src_base.display());
    let inputs = common_inputs_stamp(config);
    collect_tests_from_dir(config, &config.src_base, &PathBuf::new(), &inputs, tests)
        .unwrap_or_else(|_| panic!("Could not read tests from {}", config.src_base.display()));
}

/// Returns a stamp constructed from input files common to all test cases.
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

    stamp.add_dir(&config.run_lib_path);

    if let Some(ref rustdoc_path) = config.rustdoc_path {
        stamp.add_path(&rustdoc_path);
        stamp.add_path(&rust_src_dir.join("src/etc/htmldocck.py"));
    }

    // Compiletest itself.
    stamp.add_dir(&rust_src_dir.join("src/tools/compiletest/"));

    stamp
}

fn collect_tests_from_dir(
    config: &Config,
    dir: &Path,
    relative_dir_path: &Path,
    inputs: &Stamp,
    tests: &mut Vec<test::TestDescAndFn>,
) -> io::Result<()> {
    // Ignore directories that contain a file named `compiletest-ignore-dir`.
    if dir.join("compiletest-ignore-dir").exists() {
        return Ok(());
    }

    if config.mode == Mode::RunMake && dir.join("Makefile").exists() {
        let paths = TestPaths {
            file: dir.to_path_buf(),
            relative_dir: relative_dir_path.parent().unwrap().to_path_buf(),
        };
        tests.extend(make_test(config, &paths, inputs));
        return Ok(());
    }

    // If we find a test foo/bar.rs, we have to build the
    // output directory `$build/foo` so we can write
    // `$build/foo/bar` into it. We do this *now* in this
    // sequential loop because otherwise, if we do it in the
    // tests themselves, they race for the privilege of
    // creating the directories and sometimes fail randomly.
    let build_dir = output_relative_path(config, relative_dir_path);
    fs::create_dir_all(&build_dir).unwrap();

    // Add each `.rs` file as a test, and recurse further on any
    // subdirectories we find, except for `aux` directories.
    for file in fs::read_dir(dir)? {
        let file = file?;
        let file_path = file.path();
        let file_name = file.file_name();
        if is_test(&file_name) {
            debug!("found test file: {:?}", file_path.display());
            let paths =
                TestPaths { file: file_path, relative_dir: relative_dir_path.to_path_buf() };
            tests.extend(make_test(config, &paths, inputs))
        } else if file_path.is_dir() {
            let relative_file_path = relative_dir_path.join(file.file_name());
            if &file_name != "auxiliary" {
                debug!("found directory: {:?}", file_path.display());
                collect_tests_from_dir(config, &file_path, &relative_file_path, inputs, tests)?;
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

fn make_test(config: &Config, testpaths: &TestPaths, inputs: &Stamp) -> Vec<test::TestDescAndFn> {
    let early_props = if config.mode == Mode::RunMake {
        // Allow `ignore` directives to be in the Makefile.
        EarlyProps::from_file(config, &testpaths.file.join("Makefile"))
    } else {
        EarlyProps::from_file(config, &testpaths.file)
    };

    // The `should-fail` annotation doesn't apply to pretty tests,
    // since we run the pretty printer across all tests by default.
    // If desired, we could add a `should-fail-pretty` annotation.
    let should_panic = match config.mode {
        Pretty => test::ShouldPanic::No,
        _ => {
            if early_props.should_fail {
                test::ShouldPanic::Yes
            } else {
                test::ShouldPanic::No
            }
        }
    };

    // Incremental tests are special, they inherently cannot be run in parallel.
    // `runtest::run` will be responsible for iterating over revisions.
    let revisions = if early_props.revisions.is_empty() || config.mode == Mode::Incremental {
        vec![None]
    } else {
        early_props.revisions.iter().map(Some).collect()
    };
    revisions
        .into_iter()
        .map(|revision| {
            let ignore = early_props.ignore
                // Ignore tests that already run and are up to date with respect to inputs.
                || is_up_to_date(
                    config,
                    testpaths,
                    &early_props,
                    revision.map(|s| s.as_str()),
                    inputs,
                );
            test::TestDescAndFn {
                desc: test::TestDesc {
                    name: make_test_name(config, testpaths, revision),
                    ignore,
                    should_panic,
                    allow_fail: false,
                    test_type: test::TestType::Unknown,
                },
                testfn: make_test_closure(config, testpaths, revision),
            }
        })
        .collect()
}

fn stamp(config: &Config, testpaths: &TestPaths, revision: Option<&str>) -> PathBuf {
    output_base_dir(config, testpaths, revision).join("stamp")
}

fn is_up_to_date(
    config: &Config,
    testpaths: &TestPaths,
    props: &EarlyProps,
    revision: Option<&str>,
    inputs: &Stamp,
) -> bool {
    let stamp_name = stamp(config, testpaths, revision);
    // Check hash.
    let contents = match fs::read_to_string(&stamp_name) {
        Ok(f) => f,
        Err(ref e) if e.kind() == ErrorKind::InvalidData => panic!("Can't read stamp contents"),
        Err(_) => return false,
    };
    let expected_hash = runtest::compute_stamp_hash(config);
    if contents != expected_hash {
        return false;
    }

    // Check timestamps.
    let mut inputs = inputs.clone();
    // Use `add_dir` to account for run-make tests, which use their individual directory
    inputs.add_dir(&testpaths.file);

    for aux in &props.aux {
        let path = testpaths.file.parent().unwrap().join("auxiliary").join(aux);
        inputs.add_path(&path);
    }

    // UI test files.
    for extension in UI_EXTENSIONS {
        let path = &expected_output_path(testpaths, revision, &config.compare_mode, extension);
        inputs.add_path(path);
    }

    inputs < Stamp::from_path(&stamp_name)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Stamp {
    time: SystemTime,
}

impl Stamp {
    fn from_path(path: &Path) -> Self {
        let mut stamp = Stamp { time: SystemTime::UNIX_EPOCH };
        stamp.add_path(path);
        stamp
    }

    fn add_path(&mut self, path: &Path) {
        let modified = fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.time = self.time.max(modified);
    }

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

fn make_test_name(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&String>,
) -> test::TestName {
    // Convert a complete path to something like
    //
    //    ui/foo/bar/baz.rs
    let path = PathBuf::from(config.src_base.file_name().unwrap())
        .join(&testpaths.relative_dir)
        .join(&testpaths.file.file_name().unwrap());
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

fn make_test_closure(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&String>,
) -> test::TestFn {
    let config = config.clone();
    let testpaths = testpaths.clone();
    let revision = revision.cloned();
    test::DynTestFn(Box::new(move || runtest::run(config, &testpaths, revision.as_deref())))
}

/// Returns `true` if the given target is an Android target for the
/// purposes of GDB testing.
fn is_android_gdb_target(target: &str) -> bool {
    matches!(
        &target[..],
        "arm-linux-androideabi" | "armv7-linux-androideabi" | "aarch64-linux-android"
    )
}

/// Returns `true` if the given target is a MSVC target for the purpouses of CDB testing.
fn is_pc_windows_msvc_target(target: &str) -> bool {
    target.ends_with("-pc-windows-msvc")
}

fn find_cdb(target: &str) -> Option<OsString> {
    if !(cfg!(windows) && is_pc_windows_msvc_target(target)) {
        return None;
    }

    let pf86 = env::var_os("ProgramFiles(x86)").or_else(|| env::var_os("ProgramFiles"))?;
    let cdb_arch = if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else {
        return None; // No compatible CDB.exe in the Windows 10 SDK
    };

    let mut path = PathBuf::new();
    path.push(pf86);
    path.push(r"Windows Kits\10\Debuggers"); // We could check 8.1 etc. too?
    path.push(cdb_arch);
    path.push(r"cdb.exe");

    if !path.exists() {
        return None;
    }

    Some(path.into_os_string())
}

/// Returns Path to CDB
fn analyze_cdb(cdb: Option<String>, target: &str) -> (Option<OsString>, Option<[u16; 4]>) {
    let cdb = cdb.map(OsString::from).or_else(|| find_cdb(target));

    let mut version = None;
    if let Some(cdb) = cdb.as_ref() {
        if let Ok(output) = Command::new(cdb).arg("/version").output() {
            if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                version = extract_cdb_version(&first_line);
            }
        }
    }

    (cdb, version)
}

fn extract_cdb_version(full_version_line: &str) -> Option<[u16; 4]> {
    // Example full_version_line: "cdb version 10.0.18362.1"
    let version = full_version_line.rsplit(' ').next()?;
    let mut components = version.split('.');
    let major: u16 = components.next().unwrap().parse().unwrap();
    let minor: u16 = components.next().unwrap().parse().unwrap();
    let patch: u16 = components.next().unwrap_or("0").parse().unwrap();
    let build: u16 = components.next().unwrap_or("0").parse().unwrap();
    Some([major, minor, patch, build])
}

/// Returns (Path to GDB, GDB Version, GDB has Rust Support)
fn analyze_gdb(
    gdb: Option<String>,
    target: &str,
    android_cross_path: &PathBuf,
) -> (Option<String>, Option<u32>, bool) {
    #[cfg(not(windows))]
    const GDB_FALLBACK: &str = "gdb";
    #[cfg(windows)]
    const GDB_FALLBACK: &str = "gdb.exe";

    const MIN_GDB_WITH_RUST: u32 = 7011010;

    let fallback_gdb = || {
        if is_android_gdb_target(target) {
            let mut gdb_path = match android_cross_path.to_str() {
                Some(x) => x.to_owned(),
                None => panic!("cannot find android cross path"),
            };
            gdb_path.push_str("/bin/gdb");
            gdb_path
        } else {
            GDB_FALLBACK.to_owned()
        }
    };

    let gdb = match gdb {
        None => fallback_gdb(),
        Some(ref s) if s.is_empty() => fallback_gdb(), // may be empty if configure found no gdb
        Some(ref s) => s.to_owned(),
    };

    let mut version_line = None;
    if let Ok(output) = Command::new(&gdb).arg("--version").output() {
        if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
            version_line = Some(first_line.to_string());
        }
    }

    let version = match version_line {
        Some(line) => extract_gdb_version(&line),
        None => return (None, None, false),
    };

    let gdb_native_rust = version.map_or(false, |v| v >= MIN_GDB_WITH_RUST);

    (Some(gdb), version, gdb_native_rust)
}

fn extract_gdb_version(full_version_line: &str) -> Option<u32> {
    let full_version_line = full_version_line.trim();

    // GDB versions look like this: "major.minor.patch?.yyyymmdd?", with both
    // of the ? sections being optional

    // We will parse up to 3 digits for each component, ignoring the date

    // We skip text in parentheses.  This avoids accidentally parsing
    // the openSUSE version, which looks like:
    //  GNU gdb (GDB; openSUSE Leap 15.0) 8.1
    // This particular form is documented in the GNU coding standards:
    // https://www.gnu.org/prep/standards/html_node/_002d_002dversion.html#g_t_002d_002dversion

    let mut splits = full_version_line.rsplit(' ');
    let version_string = splits.next().unwrap();

    let mut splits = version_string.split('.');
    let major = splits.next().unwrap();
    let minor = splits.next().unwrap();
    let patch = splits.next();

    let major: u32 = major.parse().unwrap();
    let (minor, patch): (u32, u32) = match minor.find(not_a_digit) {
        None => {
            let minor = minor.parse().unwrap();
            let patch: u32 = match patch {
                Some(patch) => match patch.find(not_a_digit) {
                    None => patch.parse().unwrap(),
                    Some(idx) if idx > 3 => 0,
                    Some(idx) => patch[..idx].parse().unwrap(),
                },
                None => 0,
            };
            (minor, patch)
        }
        // There is no patch version after minor-date (e.g. "4-2012").
        Some(idx) => {
            let minor = minor[..idx].parse().unwrap();
            (minor, 0)
        }
    };

    Some(((major * 1000) + minor) * 1000 + patch)
}

/// Returns (LLDB version, LLDB is rust-enabled)
fn extract_lldb_version(full_version_line: &str) -> Option<(u32, bool)> {
    // Extract the major LLDB version from the given version string.
    // LLDB version strings are different for Apple and non-Apple platforms.
    // The Apple variant looks like this:
    //
    // LLDB-179.5 (older versions)
    // lldb-300.2.51 (new versions)
    //
    // We are only interested in the major version number, so this function
    // will return `Some(179)` and `Some(300)` respectively.
    //
    // Upstream versions look like:
    // lldb version 6.0.1
    //
    // There doesn't seem to be a way to correlate the Apple version
    // with the upstream version, and since the tests were originally
    // written against Apple versions, we make a fake Apple version by
    // multiplying the first number by 100.  This is a hack, but
    // normally fine because the only non-Apple version we test is
    // rust-enabled.

    let full_version_line = full_version_line.trim();

    if let Some(apple_ver) =
        full_version_line.strip_prefix("LLDB-").or_else(|| full_version_line.strip_prefix("lldb-"))
    {
        if let Some(idx) = apple_ver.find(not_a_digit) {
            let version: u32 = apple_ver[..idx].parse().unwrap();
            return Some((version, full_version_line.contains("rust-enabled")));
        }
    } else if let Some(lldb_ver) = full_version_line.strip_prefix("lldb version ") {
        if let Some(idx) = lldb_ver.find(not_a_digit) {
            let version: u32 = lldb_ver[..idx].parse().unwrap();
            return Some((version * 100, full_version_line.contains("rust-enabled")));
        }
    }
    None
}

fn not_a_digit(c: char) -> bool {
    !c.is_digit(10)
}
