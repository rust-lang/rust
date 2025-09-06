use std::collections::{BTreeSet, HashMap, HashSet};
use std::iter;
use std::process::Command;
use std::sync::OnceLock;

use build_helper::git::GitConfig;
use camino::{Utf8Path, Utf8PathBuf};
use semver::Version;

use crate::executor::ColorConfig;
use crate::fatal;
use crate::util::{Utf8PathBufExt, add_dylib_path, string_enum};

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum TestMode {
        Pretty => "pretty",
        DebugInfo => "debuginfo",
        Codegen => "codegen",
        Rustdoc => "rustdoc",
        RustdocJson => "rustdoc-json",
        CodegenUnits => "codegen-units",
        Incremental => "incremental",
        RunMake => "run-make",
        Ui => "ui",
        RustdocJs => "rustdoc-js",
        MirOpt => "mir-opt",
        Assembly => "assembly",
        CoverageMap => "coverage-map",
        CoverageRun => "coverage-run",
        Crashes => "crashes",
    }
}

impl TestMode {
    pub fn aux_dir_disambiguator(self) -> &'static str {
        // Pretty-printing tests could run concurrently, and if they do,
        // they need to keep their output segregated.
        match self {
            TestMode::Pretty => ".pretty",
            _ => "",
        }
    }

    pub fn output_dir_disambiguator(self) -> &'static str {
        // Coverage tests use the same test files for multiple test modes,
        // so each mode should have a separate output directory.
        match self {
            TestMode::CoverageMap | TestMode::CoverageRun => self.to_str(),
            _ => "",
        }
    }
}

// Note that coverage tests use the same test files for multiple test modes.
string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum TestSuite {
        AssemblyLlvm => "assembly-llvm",
        CodegenLlvm => "codegen-llvm",
        CodegenUnits => "codegen-units",
        Coverage => "coverage",
        CoverageRunRustdoc => "coverage-run-rustdoc",
        Crashes => "crashes",
        Debuginfo => "debuginfo",
        Incremental => "incremental",
        MirOpt => "mir-opt",
        Pretty => "pretty",
        RunMake => "run-make",
        RunMakeCargo => "run-make-cargo",
        Rustdoc => "rustdoc",
        RustdocGui => "rustdoc-gui",
        RustdocJs => "rustdoc-js",
        RustdocJsStd=> "rustdoc-js-std",
        RustdocJson => "rustdoc-json",
        RustdocUi => "rustdoc-ui",
        Ui => "ui",
        UiFullDeps => "ui-fulldeps",
    }
}

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug, Hash)]
    pub enum PassMode {
        Check => "check",
        Build => "build",
        Run => "run",
    }
}

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug, Hash)]
    pub enum RunResult {
        Pass => "run-pass",
        Fail => "run-fail",
        Crash => "run-crash",
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum RunFailMode {
    /// Running the program must make it exit with a regular failure exit code
    /// in the range `1..=127`. If the program is terminated by e.g. a signal
    /// the test will fail.
    Fail,
    /// Running the program must result in a crash, e.g. by `SIGABRT` or
    /// `SIGSEGV` on Unix or on Windows by having an appropriate NTSTATUS high
    /// bit in the exit code.
    Crash,
    /// Running the program must either fail or crash. Useful for e.g. sanitizer
    /// tests since some sanitizer implementations exit the process with code 1
    /// to in the face of memory errors while others abort (crash) the process
    /// in the face of memory errors.
    FailOrCrash,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FailMode {
    Check,
    Build,
    Run(RunFailMode),
}

string_enum! {
    #[derive(Clone, Debug, PartialEq)]
    pub enum CompareMode {
        Polonius => "polonius",
        NextSolver => "next-solver",
        NextSolverCoherence => "next-solver-coherence",
        SplitDwarf => "split-dwarf",
        SplitDwarfSingle => "split-dwarf-single",
    }
}

string_enum! {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Debugger {
        Cdb => "cdb",
        Gdb => "gdb",
        Lldb => "lldb",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PanicStrategy {
    #[default]
    Unwind,
    Abort,
}

impl PanicStrategy {
    pub(crate) fn for_miropt_test_tools(&self) -> miropt_test_tools::PanicStrategy {
        match self {
            PanicStrategy::Unwind => miropt_test_tools::PanicStrategy::Unwind,
            PanicStrategy::Abort => miropt_test_tools::PanicStrategy::Abort,
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sanitizer {
    Address,
    Cfi,
    Dataflow,
    Kcfi,
    KernelAddress,
    Leak,
    Memory,
    Memtag,
    Safestack,
    ShadowCallStack,
    Thread,
    Hwaddress,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CodegenBackend {
    Cranelift,
    Gcc,
    Llvm,
}

impl<'a> TryFrom<&'a str> for CodegenBackend {
    type Error = &'static str;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "cranelift" => Ok(Self::Cranelift),
            "gcc" => Ok(Self::Gcc),
            "llvm" => Ok(Self::Llvm),
            _ => Err("unknown backend"),
        }
    }
}

impl CodegenBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cranelift => "cranelift",
            Self::Gcc => "gcc",
            Self::Llvm => "llvm",
        }
    }
}

/// Configuration for `compiletest` *per invocation*.
///
/// In terms of `bootstrap`, this means that `./x test tests/ui tests/run-make` actually correspond
/// to *two* separate invocations of `compiletest`.
///
/// FIXME: this `Config` struct should be broken up into smaller logically contained sub-config
/// structs, it's too much of a "soup" of everything at the moment.
///
/// # Configuration sources
///
/// Configuration values for `compiletest` comes from several sources:
///
/// - CLI args passed from `bootstrap` while running the `compiletest` binary.
/// - Env vars.
/// - Discovery (e.g. trying to identify a suitable debugger based on filesystem discovery).
/// - Cached output of running the `rustc` under test (e.g. output of `rustc` print requests).
///
/// FIXME: make sure we *clearly* account for sources of *all* config options.
///
/// FIXME: audit these options to make sure we are not hashing less than necessary for build stamp
/// (for changed test detection).
#[derive(Debug, Clone)]
pub struct Config {
    /// Some [`TestMode`]s support [snapshot testing], where a *reference snapshot* of outputs (of
    /// `stdout`, `stderr`, or other form of artifacts) can be compared to the *actual output*.
    ///
    /// This option can be set to `true` to update the *reference snapshots* in-place, otherwise
    /// `compiletest` will only try to compare.
    ///
    /// [snapshot testing]: https://jestjs.io/docs/snapshot-testing
    pub bless: bool,

    /// Attempt to stop as soon as possible after any test fails. We may still run a few more tests
    /// before stopping when multiple test threads are used.
    pub fail_fast: bool,

    /// Path to libraries needed to run the *staged* `rustc`-under-test on the **host** platform.
    ///
    /// FIXME: maybe rename this to reflect (1) which target platform (host, not target), and (2)
    /// which `rustc` (the `rustc`-under-test, not the stage 0 `rustc` unless forced).
    pub compile_lib_path: Utf8PathBuf,

    /// Path to libraries needed to run the compiled executable for the **target** platform. This
    /// corresponds to the **target** sysroot libraries, including the **target** standard library.
    ///
    /// FIXME: maybe rename this to reflect (1) which target platform (target, not host), and (2)
    /// what "run libraries" are against.
    ///
    /// FIXME: this is very under-documented in conjunction with the `remote-test-client` scheme and
    /// `RUNNER` scheme to actually run the target executable under the target platform environment,
    /// cf. [`Self::remote_test_client`] and [`Self::runner`].
    pub run_lib_path: Utf8PathBuf,

    /// Path to the *staged*  `rustc`-under-test. Unless forced, this `rustc` is *staged*, and must
    /// not be confused with [`Self::stage0_rustc_path`].
    ///
    /// FIXME: maybe rename this to reflect that this is the `rustc`-under-test.
    pub rustc_path: Utf8PathBuf,

    /// Path to a *staged* **host** platform cargo executable (unless stage 0 is forced). This
    /// staged `cargo` is only used within `run-make` test recipes during recipe run time (and is
    /// *not* used to compile the test recipes), and so must be staged as there may be differences
    /// between e.g. beta `cargo` vs in-tree `cargo`.
    ///
    /// FIXME: maybe rename this to reflect that this is a *staged* host cargo.
    pub cargo_path: Option<Utf8PathBuf>,

    /// Path to the stage 0 `rustc` used to build `run-make` recipes. This must not be confused with
    /// [`Self::rustc_path`].
    pub stage0_rustc_path: Option<Utf8PathBuf>,

    /// Path to the stage 1 or higher `rustc` used to obtain target information via
    /// `--print=all-target-specs-json` and similar queries.
    ///
    /// Normally this is unset, because [`Self::rustc_path`] can be used instead.
    /// But when running "stage 1" ui-fulldeps tests, `rustc_path` is a stage 0
    /// compiler, whereas target specs must be obtained from a stage 1+ compiler
    /// (in case the JSON format has changed since the last bootstrap bump).
    pub query_rustc_path: Option<Utf8PathBuf>,

    /// Path to the `rustdoc`-under-test. Like [`Self::rustc_path`], this `rustdoc` is *staged*.
    pub rustdoc_path: Option<Utf8PathBuf>,

    /// Path to the `src/tools/coverage-dump/` bootstrap tool executable.
    pub coverage_dump_path: Option<Utf8PathBuf>,

    /// Path to the Python 3 executable to use for LLDB and htmldocck.
    ///
    /// FIXME: the `lldb` setup currently requires I believe Python 3.10 **exactly**, it can't even
    /// be Python 3.11 or 3.9...
    pub python: String,

    /// Path to the `src/tools/jsondocck/` bootstrap tool executable.
    pub jsondocck_path: Option<String>,

    /// Path to the `src/tools/jsondoclint/` bootstrap tool executable.
    pub jsondoclint_path: Option<String>,

    /// Path to a host LLVM `FileCheck` executable.
    pub llvm_filecheck: Option<Utf8PathBuf>,

    /// Path to a host LLVM bintools directory.
    pub llvm_bin_dir: Option<Utf8PathBuf>,

    /// The path to the **target** `clang` executable to run `clang`-based tests with. If `None`,
    /// then these tests will be ignored.
    pub run_clang_based_tests_with: Option<String>,

    /// Path to the directory containing the sources. This corresponds to the root folder of a
    /// `rust-lang/rust` checkout.
    ///
    /// FIXME: this name is confusing, because this is actually `$checkout_root`, **not** the
    /// `$checkout_root/src/` folder.
    pub src_root: Utf8PathBuf,

    /// Path to the directory containing the test suites sources. This corresponds to the
    /// `$src_root/tests/` folder.
    ///
    /// Must be an immediate subdirectory of [`Self::src_root`].
    ///
    /// FIXME: this name is also confusing, maybe just call it `tests_root`.
    pub src_test_suite_root: Utf8PathBuf,

    /// Path to the build directory (e.g. `build/`).
    pub build_root: Utf8PathBuf,

    /// Path to the test suite specific build directory (e.g. `build/host/test/ui/`).
    ///
    /// Must be a subdirectory of [`Self::build_root`].
    pub build_test_suite_root: Utf8PathBuf,

    /// Path to the directory containing the sysroot of the `rustc`-under-test.
    ///
    /// When stage 0 is forced, this will correspond to the sysroot *of* that specified stage 0
    /// `rustc`.
    ///
    /// FIXME: this name is confusing, because it doesn't specify *which* compiler this sysroot
    /// corresponds to. It's actually the `rustc`-under-test, and not the bootstrap `rustc`, unless
    /// stage 0 is forced and no custom stage 0 `rustc` was otherwise specified (so that it
    /// *happens* to run against the bootstrap `rustc`, but this non-custom bootstrap `rustc` case
    /// is not really supported).
    pub sysroot_base: Utf8PathBuf,

    /// The number of the stage under test.
    pub stage: u32,

    /// The id of the stage under test (stage1-xxx, etc).
    ///
    /// FIXME: reconsider this string; this is hashed for test build stamp.
    pub stage_id: String,

    /// The [`TestMode`]. E.g. [`TestMode::Ui`]. Each test mode can correspond to one or more test
    /// suites.
    ///
    /// FIXME: stop using stringly-typed test suites!
    pub mode: TestMode,

    /// The test suite.
    ///
    /// Example: `tests/ui/` is [`TestSuite::Ui`] test *suite*, which happens to also be of the
    /// [`TestMode::Ui`] test *mode*.
    ///
    /// Note that the same test suite (e.g. `tests/coverage/`) may correspond to multiple test
    /// modes, e.g. `tests/coverage/` can be run under both [`TestMode::CoverageRun`] and
    /// [`TestMode::CoverageMap`].
    pub suite: TestSuite,

    /// When specified, **only** the specified [`Debugger`] will be used to run against the
    /// `tests/debuginfo` test suite. When unspecified, `compiletest` will attempt to find all three
    /// of {`lldb`, `cdb`, `gdb`} implicitly, and then try to run the `debuginfo` test suite against
    /// all three debuggers.
    ///
    /// FIXME: this implicit behavior is really nasty, in that it makes it hard for the user to
    /// control *which* debugger(s) are available and used to run the debuginfo test suite. We
    /// should have `bootstrap` allow the user to *explicitly* configure the debuggers, and *not*
    /// try to implicitly discover some random debugger from the user environment. This makes the
    /// debuginfo test suite particularly hard to work with.
    pub debugger: Option<Debugger>,

    /// Run ignored tests *unconditionally*, overriding their ignore reason.
    ///
    /// FIXME: this is wired up through the test execution logic, but **not** accessible from
    /// `bootstrap` directly; `compiletest` exposes this as `--ignored`. I.e. you'd have to use `./x
    /// test $test_suite -- --ignored=true`.
    pub run_ignored: bool,

    /// Whether *staged* `rustc`-under-test was built with debug assertions.
    ///
    /// FIXME: make it clearer that this refers to the staged `rustc`-under-test, not stage 0
    /// `rustc`.
    pub with_rustc_debug_assertions: bool,

    /// Whether *staged* `std` was built with debug assertions.
    ///
    /// FIXME: make it clearer that this refers to the staged `std`, not stage 0 `std`.
    pub with_std_debug_assertions: bool,

    /// Only run tests that match these filters (using `libtest` "test name contains" filter logic).
    ///
    /// FIXME(#139660): the current hand-rolled test executor intentionally mimics the `libtest`
    /// "test name contains" filter matching logic to preserve previous `libtest` executor behavior,
    /// but this is often not intuitive. We should consider changing that behavior with an MCP to do
    /// test path *prefix* matching which better corresponds to how `compiletest` `tests/` are
    /// organized, and how users would intuitively expect the filtering logic to work like.
    pub filters: Vec<String>,

    /// Skip tests matching these substrings. The matching logic exactly corresponds to
    /// [`Self::filters`] but inverted.
    ///
    /// FIXME(#139660): ditto on test matching behavior.
    pub skip: Vec<String>,

    /// Exactly match the filter, rather than a substring.
    ///
    /// FIXME(#139660): ditto on test matching behavior.
    pub filter_exact: bool,

    /// Force the pass mode of a check/build/run test to instead use this mode instead.
    ///
    /// FIXME: make it even more obvious (especially in PR CI where `--pass=check` is used) when a
    /// pass mode is forced when the test fails, because it can be very non-obvious when e.g. an
    /// error is emitted only when `//@ build-pass` but not `//@ check-pass`.
    pub force_pass_mode: Option<PassMode>,

    /// Explicitly enable or disable running of the target test binary.
    ///
    /// FIXME: this scheme is a bit confusing, and at times questionable. Re-evaluate this run
    /// scheme.
    ///
    /// FIXME: Currently `--run` is a tri-state, it can be `--run={auto,always,never}`, and when
    /// `--run=auto` is specified, it's run if the platform doesn't end with `-fuchsia`. See
    /// [`Config::run_enabled`].
    pub run: Option<bool>,

    /// A command line to prefix target program execution with, for running under valgrind for
    /// example, i.e. `$runner target.exe [args..]`. Similar to `CARGO_*_RUNNER` configuration.
    ///
    /// Note: this is not to be confused with [`Self::remote_test_client`], which is a different
    /// scheme.
    ///
    /// FIXME: the runner scheme is very under-documented.
    pub runner: Option<String>,

    /// Compiler flags to pass to the *staged* `rustc`-under-test when building for the **host**
    /// platform.
    pub host_rustcflags: Vec<String>,

    /// Compiler flags to pass to the *staged* `rustc`-under-test when building for the **target**
    /// platform.
    pub target_rustcflags: Vec<String>,

    /// Whether the *staged* `rustc`-under-test and the associated *staged* `std` has been built
    /// with randomized struct layouts.
    pub rust_randomized_layout: bool,

    /// Whether tests should be optimized by default (`-O`). Individual test suites and test files
    /// may override this setting.
    ///
    /// FIXME: this flag / config option is somewhat misleading. For instance, in ui tests, it's
    /// *only* applied to the [`PassMode::Run`] test crate and not its auxiliaries.
    pub optimize_tests: bool,

    /// Target platform tuple.
    pub target: String,

    /// Host platform tuple.
    pub host: String,

    /// Path to / name of the Microsoft Console Debugger (CDB) executable.
    ///
    /// FIXME: this is an *opt-in* "override" option. When this isn't provided, we try to conjure a
    /// cdb by looking at the user's program files on Windows... See `debuggers::find_cdb`.
    pub cdb: Option<Utf8PathBuf>,

    /// Version of CDB.
    ///
    /// FIXME: `cdb_version` is *derived* from cdb, but it's *not* technically a config!
    ///
    /// FIXME: audit cdb version gating.
    pub cdb_version: Option<[u16; 4]>,

    /// Path to / name of the GDB executable.
    ///
    /// FIXME: the fallback path when `gdb` isn't provided tries to find *a* `gdb` or `gdb.exe` from
    /// `PATH`, which is... arguably questionable.
    ///
    /// FIXME: we are propagating a python from `PYTHONPATH`, not from an explicit config for gdb
    /// debugger script.
    pub gdb: Option<String>,

    /// Version of GDB, encoded as ((major * 1000) + minor) * 1000 + patch
    ///
    /// FIXME: this gdb version gating scheme is possibly questionable -- gdb does not use semver,
    /// only its major version is likely materially meaningful, cf.
    /// <https://sourceware.org/gdb/wiki/Internals%20Versions>. Even the major version I'm not sure
    /// is super meaningful. Maybe min gdb `major.minor` version gating is sufficient for the
    /// purposes of debuginfo tests?
    ///
    /// FIXME: `gdb_version` is *derived* from gdb, but it's *not* technically a config!
    pub gdb_version: Option<u32>,

    /// Version of LLDB.
    ///
    /// FIXME: `lldb_version` is *derived* from lldb, but it's *not* technically a config!
    pub lldb_version: Option<u32>,

    /// Version of LLVM.
    ///
    /// FIXME: Audit the fallback derivation of
    /// [`crate::directives::extract_llvm_version_from_binary`], that seems very questionable?
    pub llvm_version: Option<Version>,

    /// Is LLVM a system LLVM.
    pub system_llvm: bool,

    /// Path to the android tools.
    ///
    /// Note: this is only used for android gdb debugger script in the debuginfo test suite.
    ///
    /// FIXME: take a look at this; this is piggy-backing off of gdb code paths but only for
    /// `arm-linux-androideabi` target.
    pub android_cross_path: Utf8PathBuf,

    /// Extra parameter to run adb on `arm-linux-androideabi`.
    ///
    /// FIXME: is this *only* `arm-linux-androideabi`, or is it also for other Tier 2/3 android
    /// targets?
    ///
    /// FIXME: take a look at this; this is piggy-backing off of gdb code paths but only for
    /// `arm-linux-androideabi` target.
    pub adb_path: String,

    /// Extra parameter to run test suite on `arm-linux-androideabi`.
    ///
    /// FIXME: is this *only* `arm-linux-androideabi`, or is it also for other Tier 2/3 android
    /// targets?
    ///
    /// FIXME: take a look at this; this is piggy-backing off of gdb code paths but only for
    /// `arm-linux-androideabi` target.
    pub adb_test_dir: String,

    /// Status whether android device available or not. When unavailable, this will cause tests to
    /// panic when the test binary is attempted to be run.
    ///
    /// FIXME: take a look at this; this also influences adb in gdb code paths in a strange way.
    pub adb_device_status: bool,

    /// Path containing LLDB's Python module.
    ///
    /// FIXME: `PYTHONPATH` takes precedence over this flag...? See `runtest::run_lldb`.
    pub lldb_python_dir: Option<String>,

    /// Verbose dump a lot of info.
    ///
    /// FIXME: this is *way* too coarse; the user can't select *which* info to verbosely dump.
    pub verbose: bool,

    /// Whether to use colors in test output.
    ///
    /// Note: the exact control mechanism is delegated to [`colored`].
    pub color: ColorConfig,

    /// Where to find the remote test client process, if we're using it.
    ///
    /// Note: this is *only* used for target platform executables created by `run-make` test
    /// recipes.
    ///
    /// Note: this is not to be confused with [`Self::runner`], which is a different scheme.
    ///
    /// FIXME: the `remote_test_client` scheme is very under-documented.
    pub remote_test_client: Option<Utf8PathBuf>,

    /// [`CompareMode`] describing what file the actual ui output will be compared to.
    ///
    /// FIXME: currently, [`CompareMode`] is a mishmash of lot of things (different borrow-checker
    /// model, different trait solver, different debugger, etc.).
    pub compare_mode: Option<CompareMode>,

    /// If true, this will generate a coverage file with UI test files that run `MachineApplicable`
    /// diagnostics but are missing `run-rustfix` annotations. The generated coverage file is
    /// created in `$test_suite_build_root/rustfix_missing_coverage.txt`
    pub rustfix_coverage: bool,

    /// Whether to run `tidy` (html-tidy) when a rustdoc test fails.
    pub has_html_tidy: bool,

    /// Whether to run `enzyme` autodiff tests.
    pub has_enzyme: bool,

    /// The current Rust channel info.
    ///
    /// FIXME: treat this more carefully; "stable", "beta" and "nightly" are definitely valid, but
    /// channel might also be "dev" or such, which should be treated as "nightly".
    pub channel: String,

    /// Whether adding git commit information such as the commit hash has been enabled for building.
    ///
    /// FIXME: `compiletest` cannot trust `bootstrap` for this information, because `bootstrap` can
    /// have bugs and had bugs on that logic. We need to figure out how to obtain this e.g. directly
    /// from CI or via git locally.
    pub git_hash: bool,

    /// The default Rust edition.
    ///
    /// FIXME: perform stronger validation for this. There are editions that *definitely* exists,
    /// but there might also be "future" edition.
    pub edition: Option<String>,

    // Configuration for various run-make tests frobbing things like C compilers or querying about
    // various LLVM component information.
    //
    // FIXME: this really should be better packaged together.
    // FIXME: these need better docs, e.g. for *host*, or for *target*?
    pub cc: String,
    pub cxx: String,
    pub cflags: String,
    pub cxxflags: String,
    pub ar: String,
    pub target_linker: Option<String>,
    pub host_linker: Option<String>,
    pub llvm_components: String,

    /// Path to a NodeJS executable. Used for JS doctests, emscripten and WASM tests.
    pub nodejs: Option<String>,
    /// Path to a npm executable. Used for rustdoc GUI tests.
    pub npm: Option<String>,

    /// Whether to rerun tests even if the inputs are unchanged.
    pub force_rerun: bool,

    /// Only rerun the tests that result has been modified according to `git status`.
    ///
    /// FIXME: this is undocumented.
    ///
    /// FIXME: how does this interact with [`Self::force_rerun`]?
    pub only_modified: bool,

    // FIXME: these are really not "config"s, but rather are information derived from
    // `rustc`-under-test. This poses an interesting conundrum: if we're testing the
    // `rustc`-under-test, can we trust its print request outputs and target cfgs? In theory, this
    // itself can break or be unreliable -- ideally, we'd be sharing these kind of information not
    // through `rustc`-under-test's execution output. In practice, however, print requests are very
    // unlikely to completely break (we also have snapshot ui tests for them). Furthermore, even if
    // we share them via some kind of static config, that static config can still be wrong! Who
    // tests the tester? Therefore, we make a pragmatic compromise here, and use information derived
    // from print requests produced by the `rustc`-under-test.
    //
    // FIXME: move them out from `Config`, because they are *not* configs.
    pub target_cfgs: OnceLock<TargetCfgs>,
    pub builtin_cfg_names: OnceLock<HashSet<String>>,
    pub supported_crate_types: OnceLock<HashSet<String>>,

    /// FIXME: this is why we still need to depend on *staged* `std`, it's because we currently rely
    /// on `#![feature(internal_output_capture)]` for [`std::io::set_output_capture`] to implement
    /// `libtest`-esque `--no-capture`.
    ///
    /// FIXME: rename this to the more canonical `no_capture`, or better, invert this to `capture`
    /// to avoid `!nocapture` double-negatives.
    pub nocapture: bool,

    /// True if the experimental new output-capture implementation should be
    /// used, avoiding the need for `#![feature(internal_output_capture)]`.
    pub new_output_capture: bool,

    /// Needed both to construct [`build_helper::git::GitConfig`].
    pub nightly_branch: String,
    pub git_merge_commit_email: String,

    /// True if the profiler runtime is enabled for this target. Used by the
    /// `needs-profiler-runtime` directive in test files.
    pub profiler_runtime: bool,

    /// Command for visual diff display, e.g. `diff-tool --color=always`.
    pub diff_command: Option<String>,

    /// Path to minicore aux library (`tests/auxiliary/minicore.rs`), used for `no_core` tests that
    /// need `core` stubs in cross-compilation scenarios that do not otherwise want/need to
    /// `-Zbuild-std`. Used in e.g. ABI tests.
    pub minicore_path: Utf8PathBuf,

    /// Current codegen backend used.
    pub default_codegen_backend: CodegenBackend,
    /// Name/path of the backend to use instead of `default_codegen_backend`.
    pub override_codegen_backend: Option<String>,
}

impl Config {
    /// Incomplete config intended for `src/tools/rustdoc-gui-test` **only** as
    /// `src/tools/rustdoc-gui-test` wants to reuse `compiletest`'s directive -> test property
    /// handling for `//@ {compile,run}-flags`, do not use for any other purpose.
    ///
    /// FIXME(#143827): this setup feels very hacky. It so happens that `tests/rustdoc-gui/`
    /// **only** uses `//@ {compile,run}-flags` for now and not any directives that actually rely on
    /// info that is assumed available in a fully populated [`Config`].
    pub fn incomplete_for_rustdoc_gui_test() -> Config {
        // FIXME(#143827): spelling this out intentionally, because this is questionable.
        //
        // For instance, `//@ ignore-stage1` will not work at all.
        Config {
            mode: TestMode::Rustdoc,
            // E.g. this has no sensible default tbh.
            suite: TestSuite::Ui,

            // Dummy values.
            edition: Default::default(),
            bless: Default::default(),
            fail_fast: Default::default(),
            compile_lib_path: Utf8PathBuf::default(),
            run_lib_path: Utf8PathBuf::default(),
            rustc_path: Utf8PathBuf::default(),
            cargo_path: Default::default(),
            stage0_rustc_path: Default::default(),
            query_rustc_path: Default::default(),
            rustdoc_path: Default::default(),
            coverage_dump_path: Default::default(),
            python: Default::default(),
            jsondocck_path: Default::default(),
            jsondoclint_path: Default::default(),
            llvm_filecheck: Default::default(),
            llvm_bin_dir: Default::default(),
            run_clang_based_tests_with: Default::default(),
            src_root: Utf8PathBuf::default(),
            src_test_suite_root: Utf8PathBuf::default(),
            build_root: Utf8PathBuf::default(),
            build_test_suite_root: Utf8PathBuf::default(),
            sysroot_base: Utf8PathBuf::default(),
            stage: Default::default(),
            stage_id: String::default(),
            debugger: Default::default(),
            run_ignored: Default::default(),
            with_rustc_debug_assertions: Default::default(),
            with_std_debug_assertions: Default::default(),
            filters: Default::default(),
            skip: Default::default(),
            filter_exact: Default::default(),
            force_pass_mode: Default::default(),
            run: Default::default(),
            runner: Default::default(),
            host_rustcflags: Default::default(),
            target_rustcflags: Default::default(),
            rust_randomized_layout: Default::default(),
            optimize_tests: Default::default(),
            target: Default::default(),
            host: Default::default(),
            cdb: Default::default(),
            cdb_version: Default::default(),
            gdb: Default::default(),
            gdb_version: Default::default(),
            lldb_version: Default::default(),
            llvm_version: Default::default(),
            system_llvm: Default::default(),
            android_cross_path: Default::default(),
            adb_path: Default::default(),
            adb_test_dir: Default::default(),
            adb_device_status: Default::default(),
            lldb_python_dir: Default::default(),
            verbose: Default::default(),
            color: Default::default(),
            remote_test_client: Default::default(),
            compare_mode: Default::default(),
            rustfix_coverage: Default::default(),
            has_html_tidy: Default::default(),
            has_enzyme: Default::default(),
            channel: Default::default(),
            git_hash: Default::default(),
            cc: Default::default(),
            cxx: Default::default(),
            cflags: Default::default(),
            cxxflags: Default::default(),
            ar: Default::default(),
            target_linker: Default::default(),
            host_linker: Default::default(),
            llvm_components: Default::default(),
            nodejs: Default::default(),
            npm: Default::default(),
            force_rerun: Default::default(),
            only_modified: Default::default(),
            target_cfgs: Default::default(),
            builtin_cfg_names: Default::default(),
            supported_crate_types: Default::default(),
            nocapture: Default::default(),
            new_output_capture: Default::default(),
            nightly_branch: Default::default(),
            git_merge_commit_email: Default::default(),
            profiler_runtime: Default::default(),
            diff_command: Default::default(),
            minicore_path: Default::default(),
            default_codegen_backend: CodegenBackend::Llvm,
            override_codegen_backend: None,
        }
    }

    /// FIXME: this run scheme is... confusing.
    pub fn run_enabled(&self) -> bool {
        self.run.unwrap_or_else(|| {
            // Auto-detect whether to run based on the platform.
            !self.target.ends_with("-fuchsia")
        })
    }

    pub fn target_cfgs(&self) -> &TargetCfgs {
        self.target_cfgs.get_or_init(|| TargetCfgs::new(self))
    }

    pub fn target_cfg(&self) -> &TargetCfg {
        &self.target_cfgs().current
    }

    pub fn matches_arch(&self, arch: &str) -> bool {
        self.target_cfg().arch == arch ||
        // Matching all the thumb variants as one can be convenient.
        // (thumbv6m, thumbv7em, thumbv7m, etc.)
        (arch == "thumb" && self.target.starts_with("thumb"))
    }

    pub fn matches_os(&self, os: &str) -> bool {
        self.target_cfg().os == os
    }

    pub fn matches_env(&self, env: &str) -> bool {
        self.target_cfg().env == env
    }

    pub fn matches_abi(&self, abi: &str) -> bool {
        self.target_cfg().abi == abi
    }

    pub fn matches_family(&self, family: &str) -> bool {
        self.target_cfg().families.iter().any(|f| f == family)
    }

    pub fn is_big_endian(&self) -> bool {
        self.target_cfg().endian == Endian::Big
    }

    pub fn get_pointer_width(&self) -> u32 {
        *&self.target_cfg().pointer_width
    }

    pub fn can_unwind(&self) -> bool {
        self.target_cfg().panic == PanicStrategy::Unwind
    }

    /// Get the list of builtin, 'well known' cfg names
    pub fn builtin_cfg_names(&self) -> &HashSet<String> {
        self.builtin_cfg_names.get_or_init(|| builtin_cfg_names(self))
    }

    /// Get the list of crate types that the target platform supports.
    pub fn supported_crate_types(&self) -> &HashSet<String> {
        self.supported_crate_types.get_or_init(|| supported_crate_types(self))
    }

    pub fn has_threads(&self) -> bool {
        // Wasm targets don't have threads unless `-threads` is in the target
        // name, such as `wasm32-wasip1-threads`.
        if self.target.starts_with("wasm") {
            return self.target.contains("threads");
        }
        true
    }

    pub fn has_asm_support(&self) -> bool {
        // This should match the stable list in `LoweringContext::lower_inline_asm`.
        static ASM_SUPPORTED_ARCHS: &[&str] = &[
            "x86",
            "x86_64",
            "arm",
            "aarch64",
            "arm64ec",
            "riscv32",
            "riscv64",
            "loongarch32",
            "loongarch64",
            "s390x",
            // These targets require an additional asm_experimental_arch feature.
            // "nvptx64", "hexagon", "mips", "mips64", "spirv", "wasm32",
        ];
        ASM_SUPPORTED_ARCHS.contains(&self.target_cfg().arch.as_str())
    }

    pub fn git_config(&self) -> GitConfig<'_> {
        GitConfig {
            nightly_branch: &self.nightly_branch,
            git_merge_commit_email: &self.git_merge_commit_email,
        }
    }

    pub fn has_subprocess_support(&self) -> bool {
        // FIXME(#135928): compiletest is always a **host** tool. Building and running an
        // capability detection executable against the **target** is not trivial. The short term
        // solution here is to hard-code some targets to allow/deny, unfortunately.

        let unsupported_target = self.target_cfg().env == "sgx"
            || matches!(self.target_cfg().arch.as_str(), "wasm32" | "wasm64")
            || self.target_cfg().os == "emscripten";
        !unsupported_target
    }
}

/// Known widths of `target_has_atomic`.
pub const KNOWN_TARGET_HAS_ATOMIC_WIDTHS: &[&str] = &["8", "16", "32", "64", "128", "ptr"];

#[derive(Debug, Clone)]
pub struct TargetCfgs {
    pub current: TargetCfg,
    pub all_targets: HashSet<String>,
    pub all_archs: HashSet<String>,
    pub all_oses: HashSet<String>,
    pub all_oses_and_envs: HashSet<String>,
    pub all_envs: HashSet<String>,
    pub all_abis: HashSet<String>,
    pub all_families: HashSet<String>,
    pub all_pointer_widths: HashSet<String>,
    pub all_rustc_abis: HashSet<String>,
}

impl TargetCfgs {
    fn new(config: &Config) -> TargetCfgs {
        let mut targets: HashMap<String, TargetCfg> = serde_json::from_str(&query_rustc_output(
            config,
            &["--print=all-target-specs-json", "-Zunstable-options"],
            Default::default(),
        ))
        .unwrap();

        let mut all_targets = HashSet::new();
        let mut all_archs = HashSet::new();
        let mut all_oses = HashSet::new();
        let mut all_oses_and_envs = HashSet::new();
        let mut all_envs = HashSet::new();
        let mut all_abis = HashSet::new();
        let mut all_families = HashSet::new();
        let mut all_pointer_widths = HashSet::new();
        // NOTE: for distinction between `abi` and `rustc_abi`, see comment on
        // `TargetCfg::rustc_abi`.
        let mut all_rustc_abis = HashSet::new();

        // If current target is not included in the `--print=all-target-specs-json` output,
        // we check whether it is a custom target from the user or a synthetic target from bootstrap.
        if !targets.contains_key(&config.target) {
            let mut envs: HashMap<String, String> = HashMap::new();

            if let Ok(t) = std::env::var("RUST_TARGET_PATH") {
                envs.insert("RUST_TARGET_PATH".into(), t);
            }

            // This returns false only when the target is neither a synthetic target
            // nor a custom target from the user, indicating it is most likely invalid.
            if config.target.ends_with(".json") || !envs.is_empty() {
                targets.insert(
                    config.target.clone(),
                    serde_json::from_str(&query_rustc_output(
                        config,
                        &[
                            "--print=target-spec-json",
                            "-Zunstable-options",
                            "--target",
                            &config.target,
                        ],
                        envs,
                    ))
                    .unwrap(),
                );
            }
        }

        for (target, cfg) in targets.iter() {
            all_archs.insert(cfg.arch.clone());
            all_oses.insert(cfg.os.clone());
            all_oses_and_envs.insert(cfg.os_and_env());
            all_envs.insert(cfg.env.clone());
            all_abis.insert(cfg.abi.clone());
            for family in &cfg.families {
                all_families.insert(family.clone());
            }
            all_pointer_widths.insert(format!("{}bit", cfg.pointer_width));
            if let Some(rustc_abi) = &cfg.rustc_abi {
                all_rustc_abis.insert(rustc_abi.clone());
            }
            all_targets.insert(target.clone());
        }

        Self {
            current: Self::get_current_target_config(config, &targets),
            all_targets,
            all_archs,
            all_oses,
            all_oses_and_envs,
            all_envs,
            all_abis,
            all_families,
            all_pointer_widths,
            all_rustc_abis,
        }
    }

    fn get_current_target_config(
        config: &Config,
        targets: &HashMap<String, TargetCfg>,
    ) -> TargetCfg {
        let mut cfg = targets[&config.target].clone();

        // To get the target information for the current target, we take the target spec obtained
        // from `--print=all-target-specs-json`, and then we enrich it with the information
        // gathered from `--print=cfg --target=$target`.
        //
        // This is done because some parts of the target spec can be overridden with `-C` flags,
        // which are respected for `--print=cfg` but not for `--print=all-target-specs-json`. The
        // code below extracts them from `--print=cfg`: make sure to only override fields that can
        // actually be changed with `-C` flags.
        for config in query_rustc_output(
            config,
            &["--print=cfg", "--target", &config.target],
            Default::default(),
        )
        .trim()
        .lines()
        {
            let (name, value) = config
                .split_once("=\"")
                .map(|(name, value)| {
                    (
                        name,
                        Some(
                            value
                                .strip_suffix('\"')
                                .expect("key-value pair should be properly quoted"),
                        ),
                    )
                })
                .unwrap_or_else(|| (config, None));

            match (name, value) {
                // Can be overridden with `-C panic=$strategy`.
                ("panic", Some("abort")) => cfg.panic = PanicStrategy::Abort,
                ("panic", Some("unwind")) => cfg.panic = PanicStrategy::Unwind,
                ("panic", other) => panic!("unexpected value for panic cfg: {other:?}"),

                ("target_has_atomic", Some(width))
                    if KNOWN_TARGET_HAS_ATOMIC_WIDTHS.contains(&width) =>
                {
                    cfg.target_has_atomic.insert(width.to_string());
                }
                ("target_has_atomic", Some(other)) => {
                    panic!("unexpected value for `target_has_atomic` cfg: {other:?}")
                }
                // Nightly-only std-internal impl detail.
                ("target_has_atomic", None) => {}
                _ => {}
            }
        }

        cfg
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TargetCfg {
    pub(crate) arch: String,
    #[serde(default = "default_os")]
    pub(crate) os: String,
    #[serde(default)]
    pub(crate) env: String,
    #[serde(default)]
    pub(crate) abi: String,
    #[serde(rename = "target-family", default)]
    pub(crate) families: Vec<String>,
    #[serde(rename = "target-pointer-width")]
    pub(crate) pointer_width: u32,
    #[serde(rename = "target-endian", default)]
    endian: Endian,
    #[serde(rename = "panic-strategy", default)]
    pub(crate) panic: PanicStrategy,
    #[serde(default)]
    pub(crate) dynamic_linking: bool,
    #[serde(rename = "supported-sanitizers", default)]
    pub(crate) sanitizers: Vec<Sanitizer>,
    #[serde(rename = "supports-xray", default)]
    pub(crate) xray: bool,
    #[serde(default = "default_reloc_model")]
    pub(crate) relocation_model: String,
    // NOTE: `rustc_abi` should not be confused with `abi`. `rustc_abi` was introduced in #137037 to
    // make SSE2 *required* by the ABI (kind of a hack to make a target feature *required* via the
    // target spec).
    pub(crate) rustc_abi: Option<String>,

    // Not present in target cfg json output, additional derived information.
    #[serde(skip)]
    /// Supported target atomic widths: e.g. `8` to `128` or `ptr`. This is derived from the builtin
    /// `target_has_atomic` `cfg`s e.g. `target_has_atomic="8"`.
    pub(crate) target_has_atomic: BTreeSet<String>,
}

impl TargetCfg {
    pub(crate) fn os_and_env(&self) -> String {
        format!("{}-{}", self.os, self.env)
    }
}

fn default_os() -> String {
    "none".into()
}

fn default_reloc_model() -> String {
    "pic".into()
}

#[derive(Eq, PartialEq, Clone, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Endian {
    #[default]
    Little,
    Big,
}

fn builtin_cfg_names(config: &Config) -> HashSet<String> {
    query_rustc_output(
        config,
        &["--print=check-cfg", "-Zunstable-options", "--check-cfg=cfg()"],
        Default::default(),
    )
    .lines()
    .map(|l| if let Some((name, _)) = l.split_once('=') { name.to_string() } else { l.to_string() })
    .chain(std::iter::once(String::from("test")))
    .collect()
}

pub const KNOWN_CRATE_TYPES: &[&str] =
    &["bin", "cdylib", "dylib", "lib", "proc-macro", "rlib", "staticlib"];

fn supported_crate_types(config: &Config) -> HashSet<String> {
    let crate_types: HashSet<_> = query_rustc_output(
        config,
        &["--target", &config.target, "--print=supported-crate-types", "-Zunstable-options"],
        Default::default(),
    )
    .lines()
    .map(|l| l.to_string())
    .collect();

    for crate_type in crate_types.iter() {
        assert!(
            KNOWN_CRATE_TYPES.contains(&crate_type.as_str()),
            "unexpected crate type `{}`: known crate types are {:?}",
            crate_type,
            KNOWN_CRATE_TYPES
        );
    }

    crate_types
}

fn query_rustc_output(config: &Config, args: &[&str], envs: HashMap<String, String>) -> String {
    let query_rustc_path = config.query_rustc_path.as_deref().unwrap_or(&config.rustc_path);

    let mut command = Command::new(query_rustc_path);
    add_dylib_path(&mut command, iter::once(&config.compile_lib_path));
    command.args(&config.target_rustcflags).args(args);
    command.env("RUSTC_BOOTSTRAP", "1");
    command.envs(envs);

    let output = match command.output() {
        Ok(output) => output,
        Err(e) => {
            fatal!("failed to run {command:?}: {e}");
        }
    };
    if !output.status.success() {
        fatal!(
            "failed to run {command:?}\n--- stdout\n{}\n--- stderr\n{}",
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

#[derive(Debug, Clone)]
pub struct TestPaths {
    pub file: Utf8PathBuf,         // e.g., compile-test/foo/bar/baz.rs
    pub relative_dir: Utf8PathBuf, // e.g., foo/bar
}

/// Used by `ui` tests to generate things like `foo.stderr` from `foo.rs`.
pub fn expected_output_path(
    testpaths: &TestPaths,
    revision: Option<&str>,
    compare_mode: &Option<CompareMode>,
    kind: &str,
) -> Utf8PathBuf {
    assert!(UI_EXTENSIONS.contains(&kind));
    let mut parts = Vec::new();

    if let Some(x) = revision {
        parts.push(x);
    }
    if let Some(ref x) = *compare_mode {
        parts.push(x.to_str());
    }
    parts.push(kind);

    let extension = parts.join(".");
    testpaths.file.with_extension(extension)
}

pub const UI_EXTENSIONS: &[&str] = &[
    UI_STDERR,
    UI_SVG,
    UI_WINDOWS_SVG,
    UI_STDOUT,
    UI_FIXED,
    UI_RUN_STDERR,
    UI_RUN_STDOUT,
    UI_STDERR_64,
    UI_STDERR_32,
    UI_STDERR_16,
    UI_COVERAGE,
    UI_COVERAGE_MAP,
];
pub const UI_STDERR: &str = "stderr";
pub const UI_SVG: &str = "svg";
pub const UI_WINDOWS_SVG: &str = "windows.svg";
pub const UI_STDOUT: &str = "stdout";
pub const UI_FIXED: &str = "fixed";
pub const UI_RUN_STDERR: &str = "run.stderr";
pub const UI_RUN_STDOUT: &str = "run.stdout";
pub const UI_STDERR_64: &str = "64bit.stderr";
pub const UI_STDERR_32: &str = "32bit.stderr";
pub const UI_STDERR_16: &str = "16bit.stderr";
pub const UI_COVERAGE: &str = "coverage";
pub const UI_COVERAGE_MAP: &str = "cov-map";

/// Absolute path to the directory where all output for all tests in the given `relative_dir` group
/// should reside. Example:
///
/// ```text
/// /path/to/build/host-tuple/test/ui/relative/
/// ```
///
/// This is created early when tests are collected to avoid race conditions.
pub fn output_relative_path(config: &Config, relative_dir: &Utf8Path) -> Utf8PathBuf {
    config.build_test_suite_root.join(relative_dir)
}

/// Generates a unique name for the test, such as `testname.revision.mode`.
pub fn output_testname_unique(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> Utf8PathBuf {
    let mode = config.compare_mode.as_ref().map_or("", |m| m.to_str());
    let debugger = config.debugger.as_ref().map_or("", |m| m.to_str());
    Utf8PathBuf::from(&testpaths.file.file_stem().unwrap())
        .with_extra_extension(config.mode.output_dir_disambiguator())
        .with_extra_extension(revision.unwrap_or(""))
        .with_extra_extension(mode)
        .with_extra_extension(debugger)
}

/// Absolute path to the directory where all output for the given
/// test/revision should reside. Example:
///   /path/to/build/host-tuple/test/ui/relative/testname.revision.mode/
pub fn output_base_dir(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> Utf8PathBuf {
    output_relative_path(config, &testpaths.relative_dir)
        .join(output_testname_unique(config, testpaths, revision))
}

/// Absolute path to the base filename used as output for the given
/// test/revision. Example:
///   /path/to/build/host-tuple/test/ui/relative/testname.revision.mode/testname
pub fn output_base_name(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> Utf8PathBuf {
    output_base_dir(config, testpaths, revision).join(testpaths.file.file_stem().unwrap())
}

/// Absolute path to the directory to use for incremental compilation. Example:
///   /path/to/build/host-tuple/test/ui/relative/testname.mode/testname.inc
pub fn incremental_dir(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> Utf8PathBuf {
    output_base_name(config, testpaths, revision).with_extension("inc")
}
