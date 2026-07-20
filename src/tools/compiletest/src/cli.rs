//! Isolates the APIs used by `bin/main.rs`, to help minimize the surface area
//! of public exports from the compiletest library crate.

use std::env;
use std::io::IsTerminal;
use std::sync::{Arc, OnceLock};

use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;

use crate::common::{
    CodegenBackend, CompareMode, Config, Debugger, ForcePassMode, TestMode, TestSuite,
};
use crate::edition::Edition;
use crate::{debuggers, directives, early_config_check, run_tests};

pub fn main() {
    tracing_subscriber::fmt::init();

    // colored checks stdout by default, but for some reason only stderr is a terminal.
    // compiletest *does* print many things to stdout, but it doesn't really matter.
    if std::io::stderr().is_terminal()
        && matches!(std::env::var("NO_COLOR").as_deref(), Err(_) | Ok("0"))
    {
        colored::control::set_override(true);
    }

    let config = Arc::new(parse_config(env::args().collect()));

    early_config_check(&config);

    run_tests(config);
}

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
    /// Build proc-macros for wasm. Assumes environment is configured to support this; e.g., std is
    /// already built appropriately.
    #[arg(long)]
    wasm_proc_macro: bool,

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

pub(crate) fn parse_config(args: Vec<String>) -> Config {
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

        wasm_proc_macro: args.wasm_proc_macro,

        jobs: args.jobs,

        parallel_frontend_threads,
        iteration_count,
    }
}
