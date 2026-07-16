#![crate_name = "compiletest"]
#![warn(unreachable_pub)]

#[cfg(test)]
mod tests;

// Public modules needed by the compiletest binary or by `rustdoc-gui-test`.
pub mod cli;
pub mod rustdoc_gui_test;

mod common;
mod debuggers;
mod diagnostics;
mod directives;
mod edition;
mod errors;
mod executor;
mod json;
mod output_capture;
mod panic_hook;
mod raise_fd_limit;
mod read2;
mod runtest;
mod util;

use core::panic;
use std::collections::HashSet;
use std::fmt::Write;
use std::io::{self, ErrorKind};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use std::{env, fs, vec};

use build_helper::git::{get_git_modified_files, get_git_untracked_files};
use camino::{Utf8Component, Utf8Path, Utf8PathBuf};
use clap::Parser;
use rayon::iter::{ParallelBridge, ParallelIterator};
use tracing::debug;
use walkdir::WalkDir;

use self::directives::{EarlyProps, make_test_description};
use crate::common::{
    CodegenBackend, CompareMode, Config, Debugger, ForcePassMode, TestMode, TestPaths, TestSuite,
    UI_EXTENSIONS, expected_output_path, output_base_dir, output_relative_path,
};
use crate::directives::{AuxProps, DirectivesCache, FileDirectives};
use crate::edition::Edition;
use crate::executor::CollectedTest;

/// Compiletest command-line arguments.
#[derive(clap::Parser)]
struct Args {
    // Required options
    /// Path to host shared libraries.
    #[arg(long)]
    compile_lib_path: Utf8PathBuf,
    /// Path to target shared libraries.
    #[arg(long)]
    run_lib_path: Utf8PathBuf,
    /// Path to rustc to use for compiling.
    #[arg(long)]
    rustc_path: Utf8PathBuf,
    /// Path to python to use for doc tests.
    #[arg(long)]
    python: String,
    /// Directory containing sources.
    #[arg(long)]
    src_root: Utf8PathBuf,
    /// Directory containing test suite sources.
    #[arg(long)]
    src_test_suite_root: Utf8PathBuf,
    /// Path to root build directory.
    #[arg(long)]
    build_root: Utf8PathBuf,
    /// Path to test suite specific build directory.
    #[arg(long)]
    build_test_suite_root: Utf8PathBuf,
    /// Directory containing the compiler sysroot.
    #[arg(long)]
    sysroot_base: Utf8PathBuf,
    /// Stage number under test.
    #[arg(long)]
    stage: u32,
    /// The target-stage identifier.
    #[arg(long)]
    stage_id: String,
    /// Which sort of compile tests to run.
    #[arg(long)]
    mode: TestMode,
    /// Which suite of compile tests to run.
    #[arg(long)]
    suite: TestSuite,
    /// Path to a C compiler.
    #[arg(long)]
    cc: String,
    /// Path to a C++ compiler.
    #[arg(long)]
    cxx: String,
    /// Flags for the C compiler.
    #[arg(long, allow_hyphen_values = true)]
    cflags: String,
    /// Flags for the CXX compiler.
    #[arg(long, allow_hyphen_values = true)]
    cxxflags: String,
    /// List of LLVM components built in.
    #[arg(long)]
    llvm_components: String,
    /// Current Rust channel.
    #[arg(long)]
    channel: String,
    /// Name of the git branch for nightly.
    #[arg(long)]
    nightly_branch: String,
    /// Email address used for finding merge commits.
    #[arg(long)]
    git_merge_commit_email: String,
    /// Path to minicore aux library.
    #[arg(long)]
    minicore_path: Utf8PathBuf,
    /// Number of parallel jobs bootstrap was configured with.
    #[arg(long)]
    jobs: u32,
    /// The host to build for.
    #[arg(long)]
    host: String,
    /// The target to build for.
    #[arg(long)]
    target: String,

    // Optional options
    /// Path to cargo to use for compiling.
    #[arg(long)]
    cargo_path: Option<Utf8PathBuf>,
    /// Path to rustc to use for compiling run-make recipes.
    #[arg(long)]
    stage0_rustc_path: Option<Utf8PathBuf>,
    /// Path to rustc to use for querying target information.
    #[arg(long)]
    query_rustc_path: Option<Utf8PathBuf>,
    /// Path to rustdoc to use for compiling.
    #[arg(long)]
    rustdoc_path: Option<Utf8PathBuf>,
    /// Path to coverage-dump to use in tests.
    #[arg(long)]
    coverage_dump_path: Option<Utf8PathBuf>,
    /// Path to jsondocck to use for doc tests.
    #[arg(long)]
    jsondocck_path: Option<Utf8PathBuf>,
    /// Path to jsondoclint to use for doc tests.
    #[arg(long)]
    jsondoclint_path: Option<Utf8PathBuf>,
    /// Path to Clang executable.
    #[arg(long)]
    run_clang_based_tests_with: Option<Utf8PathBuf>,
    /// Path to LLVM's FileCheck binary.
    #[arg(long)]
    llvm_filecheck: Option<Utf8PathBuf>,
    /// Path to LLVM's bin directory.
    #[arg(long)]
    llvm_bin_dir: Option<Utf8PathBuf>,
    /// The name of nodejs.
    #[arg(long)]
    nodejs: Option<Utf8PathBuf>,
    /// The name of npm.
    #[arg(long)]
    npm: Option<Utf8PathBuf>,
    /// Path to the remote test client.
    #[arg(long)]
    remote_test_client: Option<Utf8PathBuf>,
    /// Path to CDB to use for CDB debuginfo tests.
    #[arg(long)]
    cdb: Option<Utf8PathBuf>,
    /// Path to GDB to use for GDB debuginfo tests.
    #[arg(long)]
    gdb: Option<Utf8PathBuf>,
    /// Path to LLDB to use for LLDB debuginfo tests.
    #[arg(long)]
    lldb: Option<Utf8PathBuf>,
    /// The version of LLDB used.
    #[arg(long)]
    lldb_version: Option<String>,
    /// The version of LLVM used.
    #[arg(long)]
    llvm_version: Option<String>,
    /// Android NDK standalone path.
    #[arg(long)]
    android_cross_path: Option<Utf8PathBuf>,
    /// Path to the android debugger.
    #[arg(long)]
    adb_path: Option<Utf8PathBuf>,
    /// Path to tests for the android debugger.
    #[arg(long)]
    adb_test_dir: Option<Utf8PathBuf>,
    /// Path to an archiver.
    #[arg(long, default_value = "ar")]
    ar: String,
    /// Path to a linker for the target.
    #[arg(long)]
    target_linker: Option<String>,
    /// Path to a linker for the host.
    #[arg(long)]
    host_linker: Option<String>,
    /// Force {check,build,run}-pass tests to this mode.
    #[arg(long)]
    pass: Option<ForcePassMode>,
    /// Whether to execute run-* tests.
    #[arg(long)]
    run: Option<String>,
    /// Supervisor program to run tests under (eg. emulator, valgrind).
    #[arg(long)]
    runner: Option<String>,
    /// Mode describing what file the actual ui output will be compared to.
    #[arg(long)]
    compare_mode: Option<CompareMode>,
    /// Default Rust edition.
    #[arg(long)]
    edition: Option<Edition>,
    /// Only test a specific debugger in debuginfo tests.
    #[arg(long)]
    debugger: Option<String>,
    /// The codegen backend currently used.
    #[arg(long)]
    default_codegen_backend: Option<CodegenBackend>,
    /// The codegen backend to use instead of the default one.
    #[arg(long)]
    override_codegen_backend: Option<String>,
    /// Custom diff tool to use for displaying compiletest tests.
    #[arg(long)]
    compiletest_diff_tool: Option<String>,
    /// Number of parallel threads to use for the frontend when building test artifacts
    #[arg(long)]
    parallel_frontend_threads: Option<u32>,
    /// Number of times to execute each test.
    #[arg(long)]
    iteration_count: Option<u32>,

    // Flags
    /// Overwrite stderr/stdout files instead of complaining about a mismatch.
    #[arg(long)]
    bless: bool,
    /// Stop as soon as possible after any test fails.
    #[arg(long)]
    fail_fast: bool,
    /// Run tests marked as ignored.
    #[arg(long)]
    ignored: bool,
    /// Run tests that require enzyme.
    #[arg(long)]
    has_enzyme: bool,
    /// Run tests that require offload.
    #[arg(long)]
    has_offload: bool,
    /// Whether rustc was built with debug assertions.
    #[arg(long)]
    with_rustc_debug_assertions: bool,
    /// Whether std was built with debug assertions.
    #[arg(long)]
    with_std_debug_assertions: bool,
    /// Whether std was built with remapping.
    #[arg(long)]
    with_std_remap_debuginfo: bool,
    /// Filters match exactly.
    #[arg(long)]
    exact: bool,
    /// Set this when rustc/stdlib were compiled with randomized layouts.
    #[arg(long)]
    rust_randomized_layout: bool,
    /// Run tests with optimizations enabled.
    #[arg(long)]
    optimize_tests: bool,
    /// Run tests verbosely, showing all output.
    #[arg(long)]
    verbose: bool,
    /// Show verbose subprocess output for successful run-make tests.
    #[arg(long)]
    verbose_run_make_subprocess_output: bool,
    /// Is LLVM the system LLVM.
    #[arg(long)]
    system_llvm: bool,
    /// Rerun tests even if the inputs are unchanged.
    #[arg(long)]
    force_rerun: bool,
    /// Only run tests that result been modified.
    #[arg(long)]
    only_modified: bool,
    // Backcompat option
    #[arg(long, hide = true)]
    nocapture: bool,
    /// Don't capture stdout/stderr of tests.
    #[arg(long)]
    no_capture: bool,
    /// Is the profiler runtime enabled for this target.
    #[arg(long)]
    profiler_runtime: bool,
    /// Run tests which rely on commit version being compiled into the binaries.
    #[arg(long)]
    git_hash: bool,
    /// Enable this to generate a Rustfix coverage file.
    #[arg(long)]
    rustfix_coverage: bool,
    /// Ignore `//@ ignore-backends` directives.
    #[arg(long)]
    bypass_ignore_backends: bool,

    // These values can be entered multiple times, for example:
    // --skip foo --skip bar
    /// Skip tests matching SUBSTRING.
    #[arg(long)]
    skip: Vec<String>,
    /// Flags to pass to rustc for host.
    #[arg(long, allow_hyphen_values = true)]
    host_rustcflags: Vec<String>,
    /// Flags to pass to rustc for target.
    #[arg(long, allow_hyphen_values = true)]
    target_rustcflags: Vec<String>,

    // Positional arguments
    /// Test name filters.
    /// All leftover arguments will be stored in this list.
    filters: Vec<String>,
}

fn parse_config(args: Vec<String>) -> Config {
    let args = Args::parse_from(args);

    fn make_absolute(path: Utf8PathBuf) -> Utf8PathBuf {
        if path.is_relative() {
            Utf8PathBuf::try_from(env::current_dir().unwrap()).unwrap().join(path)
        } else {
            path
        }
    }

    if args.nocapture {
        panic!("`--nocapture` is deprecated; please use `--no-capture`");
    }

    let adb_device_status = args.target.contains("android") && args.adb_test_dir.is_some();

    // FIXME: `cdb_version` is *derived* from cdb, but it's *not* technically a config!
    let cdb_version = args.cdb.as_deref().and_then(debuggers::query_cdb_version);
    // FIXME: `gdb_version` is *derived* from gdb, but it's *not* technically a config!
    let gdb_version = args.gdb.as_deref().and_then(debuggers::query_gdb_version);
    // FIXME: `lldb_version` is *derived* from lldb, but it's *not* technically a config!
    let lldb_version = args.lldb_version.as_deref().and_then(debuggers::extract_lldb_version);
    // FIXME: this is very questionable, we really should be obtaining LLVM version info from
    // `bootstrap`, and not trying to be figuring out that in `compiletest` by running the
    // `FileCheck` binary.
    let llvm_version =
        args.llvm_version.as_deref().map(directives::extract_llvm_version).or_else(|| {
            directives::extract_llvm_version_from_binary(args.llvm_filecheck.as_ref()?.as_str())
        });

    let default_codegen_backend = args.default_codegen_backend.unwrap_or(CodegenBackend::Llvm);

    let mode = args.mode;
    let filters = if mode == TestMode::RunMake {
        args.filters
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
        args.filters.clone()
    };

    let compare_mode = args.compare_mode;

    let src_root = args.src_root;
    let src_test_suite_root = args.src_test_suite_root;
    assert!(
        src_test_suite_root.starts_with(&src_root),
        "`src-root` must be a parent of `src-test-suite-root`: `src-root`=`{}`, `src-test-suite-root` = `{}`",
        src_root,
        src_test_suite_root
    );

    let build_root = args.build_root;
    let build_test_suite_root = args.build_test_suite_root;
    assert!(build_test_suite_root.starts_with(&build_root));

    let parallel_frontend_threads =
        args.parallel_frontend_threads.unwrap_or(Config::DEFAULT_PARALLEL_FRONTEND_THREADS);
    let iteration_count = args.iteration_count.unwrap_or(Config::DEFAULT_ITERATION_COUNT);
    assert!(iteration_count > 0, "`--iteration-count` must be a positive integer");

    Config {
        bless: args.bless,
        fail_fast: args.fail_fast || env::var_os("RUSTC_TEST_FAIL_FAST").is_some(),

        host_compile_lib_path: make_absolute(args.compile_lib_path),
        target_run_lib_path: make_absolute(args.run_lib_path),
        rustc_path: args.rustc_path,
        cargo_path: args.cargo_path,
        stage0_rustc_path: args.stage0_rustc_path,
        query_rustc_path: args.query_rustc_path,
        rustdoc_path: args.rustdoc_path,
        coverage_dump_path: args.coverage_dump_path,
        python: args.python,
        jsondocck_path: args.jsondocck_path,
        jsondoclint_path: args.jsondoclint_path,
        run_clang_based_tests_with: args.run_clang_based_tests_with,
        llvm_filecheck: args.llvm_filecheck,
        llvm_bin_dir: args.llvm_bin_dir,

        src_root,
        src_test_suite_root,

        build_root,
        build_test_suite_root,

        sysroot_base: args.sysroot_base,

        stage: args.stage,
        stage_id: args.stage_id,

        mode,
        suite: args.suite,
        debugger: args.debugger.map(|debugger| {
            debugger
                .parse::<Debugger>()
                .unwrap_or_else(|_| panic!("unknown `--debugger` option `{debugger}` given"))
        }),
        run_ignored: args.ignored,
        with_rustc_debug_assertions: args.with_rustc_debug_assertions,
        with_std_debug_assertions: args.with_std_debug_assertions,
        with_std_remap_debuginfo: args.with_std_remap_debuginfo,
        filters,
        skip: args.skip,
        filter_exact: args.exact,
        force_pass_mode: args.pass,
        // FIXME: this run scheme is... confusing.
        run: args.run.and_then(|mode| match mode.as_str() {
            "auto" => None,
            "always" => Some(true),
            "never" => Some(false),
            _ => panic!("unknown `--run` option `{}` given", mode),
        }),
        runner: args.runner,
        host_rustcflags: args.host_rustcflags,
        target_rustcflags: args.target_rustcflags,
        optimize_tests: args.optimize_tests,
        rust_randomized_layout: args.rust_randomized_layout,
        target: args.target,
        host: args.host,
        cdb: args.cdb,
        cdb_version,
        gdb: args.gdb,
        gdb_version,
        lldb: args.lldb,
        lldb_version,
        llvm_version,
        system_llvm: args.system_llvm,
        android_cross_path: args.android_cross_path,
        adb_path: args.adb_path,
        adb_test_dir: args.adb_test_dir,
        adb_device_status,
        verbose: args.verbose,
        verbose_run_make_subprocess_output: args.verbose_run_make_subprocess_output,
        only_modified: args.only_modified,
        remote_test_client: args.remote_test_client,
        compare_mode,
        rustfix_coverage: args.rustfix_coverage,
        has_enzyme: args.has_enzyme,
        has_offload: args.has_offload,
        channel: args.channel,
        git_hash: args.git_hash,
        edition: args.edition,

        cc: args.cc,
        cxx: args.cxx,
        cflags: args.cflags,
        cxxflags: args.cxxflags,
        ar: args.ar,
        target_linker: args.target_linker,
        host_linker: args.host_linker,
        llvm_components: args.llvm_components,
        nodejs: args.nodejs,

        force_rerun: args.force_rerun,

        target_cfgs: OnceLock::new(),
        builtin_cfg_names: OnceLock::new(),
        supported_crate_types: OnceLock::new(),

        capture: !args.no_capture,

        nightly_branch: args.nightly_branch,
        git_merge_commit_email: args.git_merge_commit_email,

        profiler_runtime: args.profiler_runtime,

        diff_command: args.compiletest_diff_tool,

        minicore_path: args.minicore_path,

        default_codegen_backend,
        override_codegen_backend: args.override_codegen_backend,
        bypass_ignore_backends: args.bypass_ignore_backends,

        jobs: args.jobs,

        parallel_frontend_threads,
        iteration_count,
    }
}

/// Called by `main` after the config has been parsed.
fn run_tests(config: Arc<Config>) {
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

    let mut configs = Vec::new();
    if let TestMode::DebugInfo = config.mode {
        // Debugging emscripten code doesn't make sense today
        if !config.target.contains("emscripten") {
            // FIXME: ideally, we would just have one config, and then have some mechanism of
            // generating multiple variants of a test, one for each debugger (something like
            // debuginfo revisions). But for now, we just create three configs.
            configs.extend([
                Arc::new(Config { debugger: Some(Debugger::Cdb), ..config.as_ref().clone() }),
                Arc::new(Config { debugger: Some(Debugger::Gdb), ..config.as_ref().clone() }),
                Arc::new(Config { debugger: Some(Debugger::Lldb), ..config.as_ref().clone() }),
            ]);

            // FIXME: this should ideally happen somewhere else..
            if config.target.contains("android") {
                println!("{} debug-info test uses tcp 5039 port. please reserve it", config.target);

                // android debug-info test uses remote debugger so, we test 1 thread
                // at once as they're all sharing the same TCP port to communicate
                // over.
                //
                // we should figure out how to lift this restriction! (run them all
                // on different ports allocated dynamically).
                //
                // SAFETY: at this point we are still single-threaded.
                unsafe { env::set_var("RUST_TEST_THREADS", "1") };
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
fn collect_and_make_tests(config: Arc<Config>) -> Vec<CollectedTest> {
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
        "src/etc/lldb_batchmode",
        "src/etc/lldb_lookup.py",
        "src/etc/lldb_providers.py",
    ];
    for file in &pretty_printer_files {
        let path = src_root.join(file);
        stamp.add_path(&path);
    }

    stamp.add_dir(&src_root.join("src/etc/natvis"));

    stamp.add_dir(&config.target_run_lib_path);

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
        && let Ok(backend) = backend.parse::<CodegenBackend>()
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
fn is_test(file_name: &str) -> bool {
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
    let file_contents =
        fs::read_to_string(&test_path).expect("reading test file for directives should succeed");
    let file_directives = FileDirectives::from_file_contents(&test_path, &file_contents);

    if let Err(message) = directives::do_early_directives_check(cx.config.mode, &file_directives) {
        // FIXME(Zalathar): Overhaul compiletest error handling so that we
        // don't have to resort to ad-hoc panics everywhere.
        panic!("directives check failed:\n{message}");
    }
    let early_props = EarlyProps::from_file_directives(&cx.config, &file_directives);

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
        let (test_name, filterable_path) =
            make_test_name_and_filterable_path(&cx.config, testpaths, revision);

        // While scanning for ignore/only/needs directives, also collect aux
        // paths for up-to-date checking.
        let mut aux_props = AuxProps::default();

        // Create a description struct for the test/revision.
        // This is where `ignore-*`/`only-*`/`needs-*` directives are handled,
        // because they historically needed to set the libtest ignored flag.
        let mut desc = make_test_description(
            &cx.config,
            &cx.cache,
            test_name,
            &test_path,
            &filterable_path,
            &file_directives,
            revision,
            &mut collector.poisoned,
            &mut aux_props,
        );

        // If a test's inputs haven't changed since the last time it ran,
        // mark it as ignored so that the executor will skip it.
        if !desc.is_ignored()
            && !cx.config.force_rerun
            && is_up_to_date(cx, testpaths, &aux_props, revision)
        {
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
    aux_props: &AuxProps,
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

    for aux in aux_props.all_aux_path_strings() {
        // FIXME(Zalathar): Perform all `auxiliary` path resolution in one place.
        // FIXME(Zalathar): This only finds auxiliary files used _directly_ by
        // the test file; if a transitive auxiliary is modified, the test might
        // be treated as "up-to-date" even though it should run.
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
    aux_props: &AuxProps,
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
    for path in files_related_to_test(&cx.config, testpaths, aux_props, revision) {
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
/// - `tests/rustdoc-html/primitive.rs`
/// - `tests/rustdoc-html/primitive/no_std.rs`
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

fn early_config_check(config: &Config) {
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
