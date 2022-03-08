pub use self::Mode::*;

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::util::PathBufExt;
use test::ColorConfig;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    RunPassValgrind,
    Pretty,
    DebugInfo,
    Codegen,
    Rustdoc,
    RustdocJson,
    CodegenUnits,
    Incremental,
    RunMake,
    Ui,
    JsDocTest,
    MirOpt,
    Assembly,
}

impl Mode {
    pub fn disambiguator(self) -> &'static str {
        // Pretty-printing tests could run concurrently, and if they do,
        // they need to keep their output segregated.
        match self {
            Pretty => ".pretty",
            _ => "",
        }
    }
}

impl FromStr for Mode {
    type Err = ();
    fn from_str(s: &str) -> Result<Mode, ()> {
        match s {
            "run-pass-valgrind" => Ok(RunPassValgrind),
            "pretty" => Ok(Pretty),
            "debuginfo" => Ok(DebugInfo),
            "codegen" => Ok(Codegen),
            "rustdoc" => Ok(Rustdoc),
            "rustdoc-json" => Ok(RustdocJson),
            "codegen-units" => Ok(CodegenUnits),
            "incremental" => Ok(Incremental),
            "run-make" => Ok(RunMake),
            "ui" => Ok(Ui),
            "js-doc-test" => Ok(JsDocTest),
            "mir-opt" => Ok(MirOpt),
            "assembly" => Ok(Assembly),
            _ => Err(()),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match *self {
            RunPassValgrind => "run-pass-valgrind",
            Pretty => "pretty",
            DebugInfo => "debuginfo",
            Codegen => "codegen",
            Rustdoc => "rustdoc",
            RustdocJson => "rustdoc-json",
            CodegenUnits => "codegen-units",
            Incremental => "incremental",
            RunMake => "run-make",
            Ui => "ui",
            JsDocTest => "js-doc-test",
            MirOpt => "mir-opt",
            Assembly => "assembly",
        };
        fmt::Display::fmt(s, f)
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Hash)]
pub enum PassMode {
    Check,
    Build,
    Run,
}

impl FromStr for PassMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "check" => Ok(PassMode::Check),
            "build" => Ok(PassMode::Build),
            "run" => Ok(PassMode::Run),
            _ => Err(()),
        }
    }
}

impl fmt::Display for PassMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match *self {
            PassMode::Check => "check",
            PassMode::Build => "build",
            PassMode::Run => "run",
        };
        fmt::Display::fmt(s, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FailMode {
    Check,
    Build,
    Run,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompareMode {
    Nll,
    Polonius,
    Chalk,
    SplitDwarf,
    SplitDwarfSingle,
}

impl CompareMode {
    pub(crate) fn to_str(&self) -> &'static str {
        match *self {
            CompareMode::Nll => "nll",
            CompareMode::Polonius => "polonius",
            CompareMode::Chalk => "chalk",
            CompareMode::SplitDwarf => "split-dwarf",
            CompareMode::SplitDwarfSingle => "split-dwarf-single",
        }
    }

    pub fn parse(s: String) -> CompareMode {
        match s.as_str() {
            "nll" => CompareMode::Nll,
            "polonius" => CompareMode::Polonius,
            "chalk" => CompareMode::Chalk,
            "split-dwarf" => CompareMode::SplitDwarf,
            "split-dwarf-single" => CompareMode::SplitDwarfSingle,
            x => panic!("unknown --compare-mode option: {}", x),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Debugger {
    Cdb,
    Gdb,
    Lldb,
}

impl Debugger {
    fn to_str(&self) -> &'static str {
        match self {
            Debugger::Cdb => "cdb",
            Debugger::Gdb => "gdb",
            Debugger::Lldb => "lldb",
        }
    }
}

impl fmt::Display for Debugger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.to_str(), f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PanicStrategy {
    Unwind,
    Abort,
}

/// Configuration for compiletest
#[derive(Debug, Clone)]
pub struct Config {
    /// `true` to overwrite stderr/stdout files instead of complaining about changes in output.
    pub bless: bool,

    /// The library paths required for running the compiler.
    pub compile_lib_path: PathBuf,

    /// The library paths required for running compiled programs.
    pub run_lib_path: PathBuf,

    /// The rustc executable.
    pub rustc_path: PathBuf,

    /// The rustdoc executable.
    pub rustdoc_path: Option<PathBuf>,

    /// The rust-demangler executable.
    pub rust_demangler_path: Option<PathBuf>,

    /// The Python executable to use for LLDB.
    pub lldb_python: String,

    /// The Python executable to use for htmldocck.
    pub docck_python: String,

    /// The jsondocck executable.
    pub jsondocck_path: Option<String>,

    /// The LLVM `FileCheck` binary path.
    pub llvm_filecheck: Option<PathBuf>,

    /// Path to LLVM's bin directory.
    pub llvm_bin_dir: Option<PathBuf>,

    /// The valgrind path.
    pub valgrind_path: Option<String>,

    /// Whether to fail if we can't run run-pass-valgrind tests under valgrind
    /// (or, alternatively, to silently run them like regular run-pass tests).
    pub force_valgrind: bool,

    /// The path to the Clang executable to run Clang-based tests with. If
    /// `None` then these tests will be ignored.
    pub run_clang_based_tests_with: Option<String>,

    /// The directory containing the tests to run
    pub src_base: PathBuf,

    /// The directory where programs should be built
    pub build_base: PathBuf,

    /// The name of the stage being built (stage1, etc)
    pub stage_id: String,

    /// The test mode, e.g. ui or debuginfo.
    pub mode: Mode,

    /// The test suite (essentially which directory is running, but without the
    /// directory prefix such as src/test)
    pub suite: String,

    /// The debugger to use in debuginfo mode. Unset otherwise.
    pub debugger: Option<Debugger>,

    /// Run ignored tests
    pub run_ignored: bool,

    /// Only run tests that match these filters
    pub filters: Vec<String>,

    /// Exactly match the filter, rather than a substring
    pub filter_exact: bool,

    /// Force the pass mode of a check/build/run-pass test to this mode.
    pub force_pass_mode: Option<PassMode>,

    /// Explicitly enable or disable running.
    pub run: Option<bool>,

    /// Write out a parseable log of tests that were run
    pub logfile: Option<PathBuf>,

    /// A command line to prefix program execution with,
    /// for running under valgrind
    pub runtool: Option<String>,

    /// Flags to pass to the compiler when building for the host
    pub host_rustcflags: Option<String>,

    /// Flags to pass to the compiler when building for the target
    pub target_rustcflags: Option<String>,

    /// What panic strategy the target is built with.  Unwind supports Abort, but
    /// not vice versa.
    pub target_panic: PanicStrategy,

    /// Target system to be tested
    pub target: String,

    /// Host triple for the compiler being invoked
    pub host: String,

    /// Path to / name of the Microsoft Console Debugger (CDB) executable
    pub cdb: Option<OsString>,

    /// Version of CDB
    pub cdb_version: Option<[u16; 4]>,

    /// Path to / name of the GDB executable
    pub gdb: Option<String>,

    /// Version of GDB, encoded as ((major * 1000) + minor) * 1000 + patch
    pub gdb_version: Option<u32>,

    /// Whether GDB has native rust support
    pub gdb_native_rust: bool,

    /// Version of LLDB
    pub lldb_version: Option<u32>,

    /// Whether LLDB has native rust support
    pub lldb_native_rust: bool,

    /// Version of LLVM
    pub llvm_version: Option<u32>,

    /// Is LLVM a system LLVM
    pub system_llvm: bool,

    /// Path to the android tools
    pub android_cross_path: PathBuf,

    /// Extra parameter to run adb on arm-linux-androideabi
    pub adb_path: String,

    /// Extra parameter to run test suite on arm-linux-androideabi
    pub adb_test_dir: String,

    /// status whether android device available or not
    pub adb_device_status: bool,

    /// the path containing LLDB's Python module
    pub lldb_python_dir: Option<String>,

    /// Explain what's going on
    pub verbose: bool,

    /// Print one character per test instead of one line
    pub quiet: bool,

    /// Whether to use colors in test.
    pub color: ColorConfig,

    /// where to find the remote test client process, if we're using it
    pub remote_test_client: Option<PathBuf>,

    /// mode describing what file the actual ui output will be compared to
    pub compare_mode: Option<CompareMode>,

    /// If true, this will generate a coverage file with UI test files that run `MachineApplicable`
    /// diagnostics but are missing `run-rustfix` annotations. The generated coverage file is
    /// created in `/<build_base>/rustfix_missing_coverage.txt`
    pub rustfix_coverage: bool,

    /// whether to run `tidy` when a rustdoc test fails
    pub has_tidy: bool,

    /// The current Rust channel
    pub channel: String,

    /// The default Rust edition
    pub edition: Option<String>,

    // Configuration for various run-make tests frobbing things like C compilers
    // or querying about various LLVM component information.
    pub cc: String,
    pub cxx: String,
    pub cflags: String,
    pub cxxflags: String,
    pub ar: String,
    pub linker: Option<String>,
    pub llvm_components: String,

    /// Path to a NodeJS executable. Used for JS doctests, emscripten and WASM tests
    pub nodejs: Option<String>,
    /// Path to a npm executable. Used for rustdoc GUI tests
    pub npm: Option<String>,

    /// Whether to rerun tests even if the inputs are unchanged.
    pub force_rerun: bool,
}

impl Config {
    pub fn run_enabled(&self) -> bool {
        self.run.unwrap_or_else(|| {
            // Auto-detect whether to run based on the platform.
            !self.target.ends_with("-fuchsia")
        })
    }
}

#[derive(Debug, Clone)]
pub struct TestPaths {
    pub file: PathBuf,         // e.g., compile-test/foo/bar/baz.rs
    pub relative_dir: PathBuf, // e.g., foo/bar
}

/// Used by `ui` tests to generate things like `foo.stderr` from `foo.rs`.
pub fn expected_output_path(
    testpaths: &TestPaths,
    revision: Option<&str>,
    compare_mode: &Option<CompareMode>,
    kind: &str,
) -> PathBuf {
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
    UI_STDOUT,
    UI_FIXED,
    UI_RUN_STDERR,
    UI_RUN_STDOUT,
    UI_STDERR_64,
    UI_STDERR_32,
    UI_STDERR_16,
];
pub const UI_STDERR: &str = "stderr";
pub const UI_STDOUT: &str = "stdout";
pub const UI_FIXED: &str = "fixed";
pub const UI_RUN_STDERR: &str = "run.stderr";
pub const UI_RUN_STDOUT: &str = "run.stdout";
pub const UI_STDERR_64: &str = "64bit.stderr";
pub const UI_STDERR_32: &str = "32bit.stderr";
pub const UI_STDERR_16: &str = "16bit.stderr";

/// Absolute path to the directory where all output for all tests in the given
/// `relative_dir` group should reside. Example:
///   /path/to/build/host-triple/test/ui/relative/
/// This is created early when tests are collected to avoid race conditions.
pub fn output_relative_path(config: &Config, relative_dir: &Path) -> PathBuf {
    config.build_base.join(relative_dir)
}

/// Generates a unique name for the test, such as `testname.revision.mode`.
pub fn output_testname_unique(
    config: &Config,
    testpaths: &TestPaths,
    revision: Option<&str>,
) -> PathBuf {
    let mode = config.compare_mode.as_ref().map_or("", |m| m.to_str());
    let debugger = config.debugger.as_ref().map_or("", |m| m.to_str());
    PathBuf::from(&testpaths.file.file_stem().unwrap())
        .with_extra_extension(revision.unwrap_or(""))
        .with_extra_extension(mode)
        .with_extra_extension(debugger)
}

/// Absolute path to the directory where all output for the given
/// test/revision should reside. Example:
///   /path/to/build/host-triple/test/ui/relative/testname.revision.mode/
pub fn output_base_dir(config: &Config, testpaths: &TestPaths, revision: Option<&str>) -> PathBuf {
    output_relative_path(config, &testpaths.relative_dir)
        .join(output_testname_unique(config, testpaths, revision))
}

/// Absolute path to the base filename used as output for the given
/// test/revision. Example:
///   /path/to/build/host-triple/test/ui/relative/testname.revision.mode/testname
pub fn output_base_name(config: &Config, testpaths: &TestPaths, revision: Option<&str>) -> PathBuf {
    output_base_dir(config, testpaths, revision).join(testpaths.file.file_stem().unwrap())
}

/// Absolute path to the directory to use for incremental compilation. Example:
///   /path/to/build/host-triple/test/ui/relative/testname.mode/testname.inc
pub fn incremental_dir(config: &Config, testpaths: &TestPaths) -> PathBuf {
    output_base_name(config, testpaths, None).with_extension("inc")
}
