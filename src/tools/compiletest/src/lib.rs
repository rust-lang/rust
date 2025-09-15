#![crate_name = "compiletest"]
// Needed by the "new" test executor that does not depend on libtest.
// FIXME(Zalathar): We should be able to get rid of `internal_output_capture`,
// by having `runtest` manually capture all of its println-like output instead.
// That would result in compiletest being written entirely in stable Rust!
#![feature(internal_output_capture)]

#[cfg(test)]
mod tests;

pub mod common;
mod debuggers;
pub mod diagnostics;
pub mod directives;
pub mod errors;
mod executor;
mod json;
mod output_capture;
mod panic_hook;
mod raise_fd_limit;
mod read2;
pub mod runtest;
pub mod util;

use core::panic;
use std::collections::HashSet;
use std::fmt::Write;
use std::io::{self, ErrorKind};
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use std::{env, fs, vec};

use build_helper::git::{get_git_modified_files, get_git_untracked_files};
use camino::{Utf8Component, Utf8Path, Utf8PathBuf};
use getopts::Options;
use rayon::iter::{ParallelBridge, ParallelIterator};
use tracing::debug;
use walkdir::WalkDir;

use self::directives::{EarlyProps, make_test_description};
use crate::common::{
    CodegenBackend, CompareMode, Config, Debugger, PassMode, TestMode, TestPaths, UI_EXTENSIONS,
    expected_output_path, output_base_dir, output_relative_path,
};
use crate::directives::DirectivesCache;
use crate::executor::{CollectedTest, ColorConfig};

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
        .optopt(
            "",
            "stage0-rustc-path",
            "path to rustc to use for compiling run-make recipes",
            "PATH",
        )
        .optopt(
            "",
            "query-rustc-path",
            "path to rustc to use for querying target information (defaults to `--rustc-path`)",
            "PATH",
        )
        .optopt("", "rustdoc-path", "path to rustdoc to use for compiling", "PATH")
        .optopt("", "coverage-dump-path", "path to coverage-dump to use in tests", "PATH")
        .reqopt("", "python", "path to python to use for doc tests", "PATH")
        .optopt("", "jsondocck-path", "path to jsondocck to use for doc tests", "PATH")
        .optopt("", "jsondoclint-path", "path to jsondoclint to use for doc tests", "PATH")
        .optopt("", "run-clang-based-tests-with", "path to Clang executable", "PATH")
        .optopt("", "llvm-filecheck", "path to LLVM's FileCheck binary", "DIR")
        .reqopt("", "src-root", "directory containing sources", "PATH")
        .reqopt("", "src-test-suite-root", "directory containing test suite sources", "PATH")
        .reqopt("", "build-root", "path to root build directory", "PATH")
        .reqopt("", "build-test-suite-root", "path to test suite specific build directory", "PATH")
        .reqopt("", "sysroot-base", "directory containing the compiler sysroot", "PATH")
        .reqopt("", "stage", "stage number under test", "N")
        .reqopt("", "stage-id", "the target-stage identifier", "stageN-TARGET")
        .reqopt(
            "",
            "mode",
            "which sort of compile tests to run",
            "pretty | debug-info | codegen | rustdoc \
            | rustdoc-json | codegen-units | incremental | run-make | ui \
            | rustdoc-js | mir-opt | assembly | crashes",
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
        .optflag("", "fail-fast", "stop as soon as possible after any test fails")
        .optopt("", "color", "coloring: auto, always, never", "WHEN")
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
            `./<build_test_suite_root>/rustfix_missing_coverage.txt`",
        )
        .optflag("", "force-rerun", "rerun tests even if the inputs are unchanged")
        .optflag("", "only-modified", "only run tests that result been modified")
        // FIXME: Temporarily retained so we can point users to `--no-capture`
        .optflag("", "nocapture", "")
        .optflag("", "no-capture", "don't capture stdout/stderr of tests")
        .optopt(
            "N",
            "new-output-capture",
            "enables or disables the new output-capture implementation",
            "off|on",
        )
        .optflag("", "profiler-runtime", "is the profiler runtime enabled for this target")
        .optflag("h", "help", "show this message")
        .reqopt("", "channel", "current Rust channel", "CHANNEL")
        .optflag(
            "",
            "git-hash",
            "run tests which rely on commit version being compiled into the binaries",
        )
        .optopt("", "edition", "default Rust edition", "EDITION")
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
        .reqopt("", "minicore-path", "path to minicore aux library", "PATH")
        .optopt(
            "",
            "debugger",
            "only test a specific debugger in debuginfo tests",
            "gdb | lldb | cdb",
        )
        .optopt(
            "",
            "default-codegen-backend",
            "the codegen backend currently used",
            "CODEGEN BACKEND NAME",
        )
        .optopt(
            "",
            "override-codegen-backend",
            "the codegen backend to use instead of the default one",
            "CODEGEN BACKEND [NAME | PATH]",
        );

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

    fn make_absolute(path: Utf8PathBuf) -> Utf8PathBuf {
        if path.is_relative() {
            Utf8PathBuf::try_from(env::current_dir().unwrap()).unwrap().join(path)
        } else {
            path
        }
    }

    fn opt_path(m: &getopts::Matches, nm: &str) -> Utf8PathBuf {
        match m.opt_str(nm) {
            Some(s) => Utf8PathBuf::from(&s),
            None => panic!("no option (=path) found for {}", nm),
        }
    }

    let target = opt_str2(matches.opt_str("target"));
    let android_cross_path = opt_path(matches, "android-cross-path");
    // FIXME: `cdb_version` is *derived* from cdb, but it's *not* technically a config!
    let (cdb, cdb_version) = debuggers::analyze_cdb(matches.opt_str("cdb"), &target);
    // FIXME: `gdb_version` is *derived* from gdb, but it's *not* technically a config!
    let (gdb, gdb_version) =
        debuggers::analyze_gdb(matches.opt_str("gdb"), &target, &android_cross_path);
    // FIXME: `lldb_version` is *derived* from lldb, but it's *not* technically a config!
    let lldb_version =
        matches.opt_str("lldb-version").as_deref().and_then(debuggers::extract_lldb_version);
    let color = match matches.opt_str("color").as_deref() {
        Some("auto") | None => ColorConfig::AutoColor,
        Some("always") => ColorConfig::AlwaysColor,
        Some("never") => ColorConfig::NeverColor,
        Some(x) => panic!("argument for --color must be auto, always, or never, but found `{}`", x),
    };
    // FIXME: this is very questionable, we really should be obtaining LLVM version info from
    // `bootstrap`, and not trying to be figuring out that in `compiletest` by running the
    // `FileCheck` binary.
    let llvm_version =
        matches.opt_str("llvm-version").as_deref().map(directives::extract_llvm_version).or_else(
            || directives::extract_llvm_version_from_binary(&matches.opt_str("llvm-filecheck")?),
        );

    let default_codegen_backend = match matches.opt_str("default-codegen-backend").as_deref() {
        Some(backend) => match CodegenBackend::try_from(backend) {
            Ok(backend) => backend,
            Err(error) => {
                panic!("invalid value `{backend}` for `--defalt-codegen-backend`: {error}")
            }
        },
        // By default, it's always llvm.
        None => CodegenBackend::Llvm,
    };
    let override_codegen_backend = matches.opt_str("override-codegen-backend");

    let run_ignored = matches.opt_present("ignored");
    let with_rustc_debug_assertions = matches.opt_present("with-rustc-debug-assertions");
    let with_std_debug_assertions = matches.opt_present("with-std-debug-assertions");
    let mode = matches.opt_str("mode").unwrap().parse().expect("invalid mode");
    let has_html_tidy = if mode == TestMode::Rustdoc {
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
    let filters = if mode == TestMode::RunMake {
        matches
            .free
            .iter()
            .map(|f| {
                // Here `f` is relative to `./tests/run-make`. So if you run
                //
                //   ./x test tests/run-make/crate-loading
                //
                //  then `f` is "crate-loading".
                let path = Utf8Path::new(f);
                let mut iter = path.iter().skip(1);

                if iter.next().is_some_and(|s| s == "rmake.rs") && iter.next().is_none() {
                    // Strip the "rmake.rs" suffix. For example, if `f` is
                    // "crate-loading/rmake.rs" then this gives us "crate-loading".
                    path.parent().unwrap().to_string()
                } else {
                    f.to_string()
                }
            })
            .collect::<Vec<_>>()
    } else {
        // Note that the filters are relative to the root dir of the different test
        // suites. For example, with:
        //
        //   ./x test tests/ui/lint/unused
        //
        // the filter is "lint/unused".
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
    if matches.opt_present("nocapture") {
        panic!("`--nocapture` is deprecated; please use `--no-capture`");
    }

    let stage = match matches.opt_str("stage") {
        Some(stage) => stage.parse::<u32>().expect("expected `--stage` to be an unsigned integer"),
        None => panic!("`--stage` is required"),
    };

    let src_root = opt_path(matches, "src-root");
    let src_test_suite_root = opt_path(matches, "src-test-suite-root");
    assert!(
        src_test_suite_root.starts_with(&src_root),
        "`src-root` must be a parent of `src-test-suite-root`: `src-root`=`{}`, `src-test-suite-root` = `{}`",
        src_root,
        src_test_suite_root
    );

    let build_root = opt_path(matches, "build-root");
    let build_test_suite_root = opt_path(matches, "build-test-suite-root");
    assert!(build_test_suite_root.starts_with(&build_root));

    Config {
        bless: matches.opt_present("bless"),
        fail_fast: matches.opt_present("fail-fast")
            || env::var_os("RUSTC_TEST_FAIL_FAST").is_some(),

        compile_lib_path: make_absolute(opt_path(matches, "compile-lib-path")),
        run_lib_path: make_absolute(opt_path(matches, "run-lib-path")),
        rustc_path: opt_path(matches, "rustc-path"),
        cargo_path: matches.opt_str("cargo-path").map(Utf8PathBuf::from),
        stage0_rustc_path: matches.opt_str("stage0-rustc-path").map(Utf8PathBuf::from),
        query_rustc_path: matches.opt_str("query-rustc-path").map(Utf8PathBuf::from),
        rustdoc_path: matches.opt_str("rustdoc-path").map(Utf8PathBuf::from),
        coverage_dump_path: matches.opt_str("coverage-dump-path").map(Utf8PathBuf::from),
        python: matches.opt_str("python").unwrap(),
        jsondocck_path: matches.opt_str("jsondocck-path"),
        jsondoclint_path: matches.opt_str("jsondoclint-path"),
        run_clang_based_tests_with: matches.opt_str("run-clang-based-tests-with"),
        llvm_filecheck: matches.opt_str("llvm-filecheck").map(Utf8PathBuf::from),
        llvm_bin_dir: matches.opt_str("llvm-bin-dir").map(Utf8PathBuf::from),

        src_root,
        src_test_suite_root,

        build_root,
        build_test_suite_root,

        sysroot_base: opt_path(matches, "sysroot-base"),

        stage,
        stage_id: matches.opt_str("stage-id").unwrap(),

        mode,
        suite: matches.opt_str("suite").unwrap().parse().expect("invalid suite"),
        debugger: matches.opt_str("debugger").map(|debugger| {
            debugger
                .parse::<Debugger>()
                .unwrap_or_else(|_| panic!("unknown `--debugger` option `{debugger}` given"))
        }),
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
        // FIXME: this run scheme is... confusing.
        run: matches.opt_str("run").and_then(|mode| match mode.as_str() {
            "auto" => None,
            "always" => Some(true),
            "never" => Some(false),
            _ => panic!("unknown `--run` option `{}` given", mode),
        }),
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
        only_modified: matches.opt_present("only-modified"),
        color,
        remote_test_client: matches.opt_str("remote-test-client").map(Utf8PathBuf::from),
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
        supported_crate_types: OnceLock::new(),

        nocapture: matches.opt_present("no-capture"),
        new_output_capture: {
            let value = matches
                .opt_str("new-output-capture")
                .or_else(|| env::var("COMPILETEST_NEW_OUTPUT_CAPTURE").ok())
                .unwrap_or_else(|| "off".to_owned());
            parse_bool_option(&value)
                .unwrap_or_else(|| panic!("unknown `--new-output-capture` value `{value}` given"))
        },

        nightly_branch: matches.opt_str("nightly-branch").unwrap(),
        git_merge_commit_email: matches.opt_str("git-merge-commit-email").unwrap(),

        profiler_runtime: matches.opt_present("profiler-runtime"),

        diff_command: matches.opt_str("compiletest-diff-tool"),

        minicore_path: opt_path(matches, "minicore-path"),

        default_codegen_backend,
        override_codegen_backend,
    }
}

/// Parses the same set of boolean values accepted by rustc command-line arguments.
///
/// Accepting all of these values is more complicated than just picking one
/// pair, but has the advantage that contributors who are used to rustc
/// shouldn't have to think about which values are legal.
fn parse_bool_option(value: &str) -> Option<bool> {
    match value {
        "off" | "no" | "n" | "false" => Some(false),
        "on" | "yes" | "y" | "true" => Some(true),
        _ => None,
    }
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
    debug!(?config, "run_tests");

    panic_hook::install_panic_hook();

    // If we want to collect rustfix coverage information,
    // we first make sure that the coverage file does not exist.
    // It will be created later on.
    if config.rustfix_coverage {
        let mut coverage_file_path = config.build_test_suite_root.clone();
        coverage_file_path.push("rustfix_missing_coverage.txt");
        if coverage_file_path.exists() {
            if let Err(e) = fs::remove_file(&coverage_file_path) {
                panic!("Could not delete {} due to {}", coverage_file_path, e)
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
    //
    // SAFETY: at this point we're still single-threaded.
    unsafe { env::set_var("__COMPAT_LAYER", "RunAsInvoker") };

    // Let tests know which target they're running as.
    //
    // SAFETY: at this point we're still single-threaded.
    unsafe { env::set_var("TARGET", &config.target) };

    let mut configs = Vec::new();
    if let TestMode::DebugInfo = config.mode {
        // Debugging emscripten code doesn't make sense today
        if !config.target.contains("emscripten") {
            match config.debugger {
                Some(Debugger::Cdb) => configs.extend(debuggers::configure_cdb(&config)),
                Some(Debugger::Gdb) => configs.extend(debuggers::configure_gdb(&config)),
                Some(Debugger::Lldb) => configs.extend(debuggers::configure_lldb(&config)),
                // FIXME: the *implicit* debugger discovery makes it really difficult to control
                // which {`cdb`, `gdb`, `lldb`} are used. These should **not** be implicitly
                // discovered by `compiletest`; these should be explicit `bootstrap` configuration
                // options that are passed to `compiletest`!
                None => {
                    configs.extend(debuggers::configure_cdb(&config));
                    configs.extend(debuggers::configure_gdb(&config));
                    configs.extend(debuggers::configure_lldb(&config));
                }
            }
        }
    } else {
        configs.push(config.clone());
    };

    // Discover all of the tests in the test suite directory, and build a `CollectedTest`
    // structure for each test (or each revision of a multi-revision test).
    let mut tests = Vec::new();
    for c in configs {
        tests.extend(collect_and_make_tests(c));
    }

    tests.sort_by(|a, b| Ord::cmp(&a.desc.name, &b.desc.name));

    // Delegate to the executor to filter and run the big list of test structures
    // created during test discovery. When the executor decides to run a test,
    // it will return control to the rest of compiletest by calling `runtest::run`.
    let ok = executor::run_tests(&config, tests);

    // Check the outcome reported by the executor.
    if !ok {
        // We want to report that the tests failed, but we also want to give
        // some indication of just what tests we were running. Especially on
        // CI, where there can be cross-compiled tests for a lot of
        // architectures, without this critical information it can be quite
        // easy to miss which tests failed, and as such fail to reproduce
        // the failure locally.

        let mut msg = String::from("Some tests failed in compiletest");
        write!(msg, " suite={}", config.suite).unwrap();

        if let Some(compare_mode) = config.compare_mode.as_ref() {
            write!(msg, " compare_mode={}", compare_mode).unwrap();
        }

        if let Some(pass_mode) = config.force_pass_mode.as_ref() {
            write!(msg, " pass_mode={}", pass_mode).unwrap();
        }

        write!(msg, " mode={}", config.mode).unwrap();
        write!(msg, " host={}", config.host).unwrap();
        write!(msg, " target={}", config.target).unwrap();

        println!("{msg}");

        std::process::exit(1);
    }
}

/// Read-only context data used during test collection.
struct TestCollectorCx {
    config: Arc<Config>,
    cache: DirectivesCache,
    common_inputs_stamp: Stamp,
    modified_tests: Vec<Utf8PathBuf>,
}

/// Mutable state used during test collection.
struct TestCollector {
    tests: Vec<CollectedTest>,
    found_path_stems: HashSet<Utf8PathBuf>,
    poisoned: bool,
}

impl TestCollector {
    fn new() -> Self {
        TestCollector { tests: vec![], found_path_stems: HashSet::new(), poisoned: false }
    }

    fn merge(&mut self, mut other: Self) {
        self.tests.append(&mut other.tests);
        self.found_path_stems.extend(other.found_path_stems);
        self.poisoned |= other.poisoned;
    }
}

/// Creates test structures for every test/revision in the test suite directory.
///
/// This always inspects _all_ test files in the suite (e.g. all 17k+ ui tests),
/// regardless of whether any filters/tests were specified on the command-line,
/// because filtering is handled later by code that was copied from libtest.
///
/// FIXME(Zalathar): Now that we no longer rely on libtest, try to overhaul
/// test discovery to take into account the filters/tests specified on the
/// command-line, instead of having to enumerate everything.
pub(crate) fn collect_and_make_tests(config: Arc<Config>) -> Vec<CollectedTest> {
    debug!("making tests from {}", config.src_test_suite_root);
    let common_inputs_stamp = common_inputs_stamp(&config);
    let modified_tests =
        modified_tests(&config, &config.src_test_suite_root).unwrap_or_else(|err| {
            fatal!("modified_tests: {}: {err}", config.src_test_suite_root);
        });
    let cache = DirectivesCache::load(&config);

    let cx = TestCollectorCx { config, cache, common_inputs_stamp, modified_tests };
    let collector = collect_tests_from_dir(&cx, &cx.config.src_test_suite_root, Utf8Path::new(""))
        .unwrap_or_else(|reason| {
            panic!("Could not read tests from {}: {reason}", cx.config.src_test_suite_root)
        });

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
    let src_root = &config.src_root;

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
        let path = src_root.join(file);
        stamp.add_path(&path);
    }

    stamp.add_dir(&src_root.join("src/etc/natvis"));

    stamp.add_dir(&config.run_lib_path);

    if let Some(ref rustdoc_path) = config.rustdoc_path {
        stamp.add_path(&rustdoc_path);
        stamp.add_path(&src_root.join("src/etc/htmldocck.py"));
    }

    // Re-run coverage tests if the `coverage-dump` tool was modified,
    // because its output format might have changed.
    if let Some(coverage_dump_path) = &config.coverage_dump_path {
        stamp.add_path(coverage_dump_path)
    }

    stamp.add_dir(&src_root.join("src/tools/run-make-support"));

    // Compiletest itself.
    stamp.add_dir(&src_root.join("src/tools/compiletest"));

    stamp
}

/// Returns a list of modified/untracked test files that should be run when
/// the `--only-modified` flag is in use.
///
/// (Might be inaccurate in some cases.)
fn modified_tests(config: &Config, dir: &Utf8Path) -> Result<Vec<Utf8PathBuf>, String> {
    // If `--only-modified` wasn't passed, the list of modified tests won't be
    // used for anything, so avoid some work and just return an empty list.
    if !config.only_modified {
        return Ok(vec![]);
    }

    let files = get_git_modified_files(
        &config.git_config(),
        Some(dir.as_std_path()),
        &vec!["rs", "stderr", "fixed"],
    )?;
    // Add new test cases to the list, it will be convenient in daily development.
    let untracked_files = get_git_untracked_files(Some(dir.as_std_path()))?.unwrap_or(vec![]);

    let all_paths = [&files[..], &untracked_files[..]].concat();
    let full_paths = {
        let mut full_paths: Vec<Utf8PathBuf> = all_paths
            .into_iter()
            .map(|f| Utf8PathBuf::from(f).with_extension("").with_extension("rs"))
            .filter_map(
                |f| if Utf8Path::new(&f).exists() { f.canonicalize_utf8().ok() } else { None },
            )
            .collect();
        full_paths.dedup();
        full_paths.sort_unstable();
        full_paths
    };
    Ok(full_paths)
}

/// Recursively scans a directory to find test files and create test structures
/// that will be handed over to the executor.
fn collect_tests_from_dir(
    cx: &TestCollectorCx,
    dir: &Utf8Path,
    relative_dir_path: &Utf8Path,
) -> io::Result<TestCollector> {
    // Ignore directories that contain a file named `compiletest-ignore-dir`.
    if dir.join("compiletest-ignore-dir").exists() {
        return Ok(TestCollector::new());
    }

    let mut components = dir.components().rev();
    if let Some(Utf8Component::Normal(last)) = components.next()
        && let Some(("assembly" | "codegen", backend)) = last.split_once('-')
        && let Some(Utf8Component::Normal(parent)) = components.next()
        && parent == "tests"
        && let Ok(backend) = CodegenBackend::try_from(backend)
        && backend != cx.config.default_codegen_backend
    {
        // We ignore asm tests which don't match the current codegen backend.
        warning!(
            "Ignoring tests in `{dir}` because they don't match the configured codegen \
             backend (`{}`)",
            cx.config.default_codegen_backend.as_str(),
        );
        return Ok(TestCollector::new());
    }

    // For run-make tests, a "test file" is actually a directory that contains an `rmake.rs`.
    if cx.config.mode == TestMode::RunMake {
        let mut collector = TestCollector::new();
        if dir.join("rmake.rs").exists() {
            let paths = TestPaths {
                file: dir.to_path_buf(),
                relative_dir: relative_dir_path.parent().unwrap().to_path_buf(),
            };
            make_test(cx, &mut collector, &paths);
            // This directory is a test, so don't try to find other tests inside it.
            return Ok(collector);
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
    fs::read_dir(dir.as_std_path())?
        .par_bridge()
        .map(|file| {
            let mut collector = TestCollector::new();
            let file = file?;
            let file_path = Utf8PathBuf::try_from(file.path()).unwrap();
            let file_name = file_path.file_name().unwrap();

            if is_test(file_name)
                && (!cx.config.only_modified || cx.modified_tests.contains(&file_path))
            {
                // We found a test file, so create the corresponding test structures.
                debug!(%file_path, "found test file");

                // Record the stem of the test file, to check for overlaps later.
                let rel_test_path = relative_dir_path.join(file_path.file_stem().unwrap());
                collector.found_path_stems.insert(rel_test_path);

                let paths =
                    TestPaths { file: file_path, relative_dir: relative_dir_path.to_path_buf() };
                make_test(cx, &mut collector, &paths);
            } else if file_path.is_dir() {
                // Recurse to find more tests in a subdirectory.
                let relative_file_path = relative_dir_path.join(file_name);
                if file_name != "auxiliary" {
                    debug!(%file_path, "found directory");
                    collector.merge(collect_tests_from_dir(cx, &file_path, &relative_file_path)?);
                }
            } else {
                debug!(%file_path, "found other file/directory");
            }
            Ok(collector)
        })
        .reduce(
            || Ok(TestCollector::new()),
            |a, b| {
                let mut a = a?;
                a.merge(b?);
                Ok(a)
            },
        )
}

/// Returns true if `file_name` looks like a proper test file name.
pub fn is_test(file_name: &str) -> bool {
    if !file_name.ends_with(".rs") {
        return false;
    }

    // `.`, `#`, and `~` are common temp-file prefixes.
    let invalid_prefixes = &[".", "#", "~"];
    !invalid_prefixes.iter().any(|p| file_name.starts_with(p))
}

/// For a single test file, creates one or more test structures (one per revision) that can be
/// handed over to the executor to run, possibly in parallel.
fn make_test(cx: &TestCollectorCx, collector: &mut TestCollector, testpaths: &TestPaths) {
    // For run-make tests, each "test file" is actually a _directory_ containing an `rmake.rs`. But
    // for the purposes of directive parsing, we want to look at that recipe file, not the directory
    // itself.
    let test_path = if cx.config.mode == TestMode::RunMake {
        testpaths.file.join("rmake.rs")
    } else {
        testpaths.file.clone()
    };

    // Scan the test file to discover its revisions, if any.
    let early_props = EarlyProps::from_file(&cx.config, &test_path);

    // Normally we create one structure per revision, with two exceptions:
    // - If a test doesn't use revisions, create a dummy revision (None) so that
    //   the test can still run.
    // - Incremental tests inherently can't run their revisions in parallel, so
    //   we treat them like non-revisioned tests here. Incremental revisions are
    //   handled internally by `runtest::run` instead.
    let revisions = if early_props.revisions.is_empty() || cx.config.mode == TestMode::Incremental {
        vec![None]
    } else {
        early_props.revisions.iter().map(|r| Some(r.as_str())).collect()
    };

    // For each revision (or the sole dummy revision), create and append a
    // `CollectedTest` that can be handed over to the test executor.
    collector.tests.extend(revisions.into_iter().map(|revision| {
        // Create a test name and description to hand over to the executor.
        let src_file = fs::File::open(&test_path).expect("open test file to parse ignores");
        let (test_name, filterable_path) =
            make_test_name_and_filterable_path(&cx.config, testpaths, revision);
        // Create a description struct for the test/revision.
        // This is where `ignore-*`/`only-*`/`needs-*` directives are handled,
        // because they historically needed to set the libtest ignored flag.
        let mut desc = make_test_description(
            &cx.config,
            &cx.cache,
            test_name,
            &test_path,
            &filterable_path,
            src_file,
            revision,
            &mut collector.poisoned,
        );

        // If a test's inputs haven't changed since the last time it ran,
        // mark it as ignored so that the executor will skip it.
        if !cx.config.force_rerun && is_up_to_date(cx, testpaths, &early_props, revision) {
            desc.ignore = true;
            // Keep this in sync with the "up-to-date" message detected by bootstrap.
            // FIXME(Zalathar): Now that we are no longer tied to libtest, we could
            // find a less fragile way to communicate this status to bootstrap.
            desc.ignore_message = Some("up-to-date".into());
        }

        let config = Arc::clone(&cx.config);
        let testpaths = testpaths.clone();
        let revision = revision.map(str::to_owned);

        CollectedTest { desc, config, testpaths, revision }
    }));
}

/// The path of the `stamp` file that gets created or updated whenever a
/// particular test completes successfully.
fn stamp_file_path(config: &Config, testpaths: &TestPaths, revision: Option<&str>) -> Utf8PathBuf {
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
) -> Vec<Utf8PathBuf> {
    let mut related = vec![];

    if testpaths.file.is_dir() {
        // run-make tests use their individual directory
        for entry in WalkDir::new(&testpaths.file) {
            let path = entry.unwrap().into_path();
            if path.is_file() {
                related.push(Utf8PathBuf::try_from(path).unwrap());
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
    related.push(config.src_root.join("tests").join("auxiliary").join("minicore.rs"));

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
    fn from_path(path: &Utf8Path) -> Self {
        let mut stamp = Stamp { time: SystemTime::UNIX_EPOCH };
        stamp.add_path(path);
        stamp
    }

    /// Updates this timestamp to the last-modified time of the specified file,
    /// if it is later than the currently-stored timestamp.
    fn add_path(&mut self, path: &Utf8Path) {
        let modified = fs::metadata(path.as_std_path())
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.time = self.time.max(modified);
    }

    /// Updates this timestamp to the most recent last-modified time of all files
    /// recursively contained in the given directory, if it is later than the
    /// currently-stored timestamp.
    fn add_dir(&mut self, path: &Utf8Path) {
        let path = path.as_std_path();
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

/// Creates a name for this test/revision that can be handed over to the executor.
fn make_test_name_and_filterable_path(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> (String, Utf8PathBuf) {
    // Print the name of the file, relative to the sources root.
    let path = testpaths.file.strip_prefix(&config.src_root).unwrap();
    let debugger = match config.debugger {
        Some(d) => format!("-{}", d),
        None => String::new(),
    };
    let mode_suffix = match config.compare_mode {
        Some(ref mode) => format!(" ({})", mode.to_str()),
        None => String::new(),
    };

    let name = format!(
        "[{}{}{}] {}{}",
        config.mode,
        debugger,
        mode_suffix,
        path,
        revision.map_or("".to_string(), |rev| format!("#{}", rev))
    );

    // `path` is the full path from the repo root like, `tests/ui/foo/bar.rs`.
    // Filtering is applied without the `tests/ui/` part, so strip that off.
    // First strip off "tests" to make sure we don't have some unexpected path.
    let mut filterable_path = path.strip_prefix("tests").unwrap().to_owned();
    // Now strip off e.g. "ui" or "run-make" component.
    filterable_path = filterable_path.components().skip(1).collect();

    (name, filterable_path)
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
fn check_for_overlapping_test_paths(found_path_stems: &HashSet<Utf8PathBuf>) {
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
            .map(|(path, check_parent)| format!("test {path} clashes with {check_parent}\n"))
            .collect();
        panic!(
            "{collisions}\n\
            Tests cannot have overlapping names. Make sure they use unique prefixes."
        );
    }
}

pub fn early_config_check(config: &Config) {
    if !config.has_html_tidy && config.mode == TestMode::Rustdoc {
        warning!("`tidy` (html-tidy.org) is not installed; diffs will not be generated");
    }

    if !config.profiler_runtime && config.mode == TestMode::CoverageRun {
        let actioned = if config.bless { "blessed" } else { "checked" };
        warning!("profiler runtime is not available, so `.coverage` files won't be {actioned}");
        help!("try setting `profiler = true` in the `[build]` section of `bootstrap.toml`");
    }

    // `RUST_TEST_NOCAPTURE` is a libtest env var, but we don't callout to libtest.
    if env::var("RUST_TEST_NOCAPTURE").is_ok() {
        warning!("`RUST_TEST_NOCAPTURE` is not supported; use the `--no-capture` flag instead");
    }
}
