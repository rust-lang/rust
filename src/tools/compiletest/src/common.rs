pub use self::Mode::*;

use std::ffi::OsString;
use std::fmt;
use std::iter;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use crate::util::{add_dylib_path, PathBufExt};
use lazycell::AtomicLazyCell;
use serde::de::{Deserialize, Deserializer, Error as _};
use std::collections::{HashMap, HashSet};
use test::{ColorConfig, OutputFormat};

macro_rules! string_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident => $repr:expr,)* }) => {
        $(#[$meta])*
        $vis enum $name {
            $($variant,)*
        }

        impl $name {
            $vis const VARIANTS: &'static [Self] = &[$(Self::$variant,)*];
            $vis const STR_VARIANTS: &'static [&'static str] = &[$(Self::$variant.to_str(),)*];

            $vis const fn to_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $repr,)*
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self.to_str(), f)
            }
        }

        impl FromStr for $name {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, ()> {
                match s {
                    $($repr => Ok(Self::$variant),)*
                    _ => Err(()),
                }
            }
        }
    }
}

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Mode {
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
        RunCoverage => "run-coverage",
    }
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Ui
    }
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

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug, Hash)]
    pub enum PassMode {
        Check => "check",
        Build => "build",
        Run => "run",
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FailMode {
    Check,
    Build,
    Run,
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

/// Configuration for compiletest
#[derive(Debug, Default, Clone)]
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

    /// The Python executable to use for LLDB and htmldocck.
    pub python: String,

    /// The jsondocck executable.
    pub jsondocck_path: Option<String>,

    /// The jsondoclint executable.
    pub jsondoclint_path: Option<String>,

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

    /// The directory containing the compiler sysroot
    pub sysroot_base: PathBuf,

    /// The name of the stage being built (stage1, etc)
    pub stage_id: String,

    /// The test mode, e.g. ui or debuginfo.
    pub mode: Mode,

    /// The test suite (essentially which directory is running, but without the
    /// directory prefix such as tests)
    pub suite: String,

    /// The debugger to use in debuginfo mode. Unset otherwise.
    pub debugger: Option<Debugger>,

    /// Run ignored tests
    pub run_ignored: bool,

    /// Only run tests that match these filters
    pub filters: Vec<String>,

    /// Skip tests tests matching these substrings. Corresponds to
    /// `test::TestOpts::skip`. `filter_exact` does not apply to these flags.
    pub skip: Vec<String>,

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
    pub host_rustcflags: Vec<String>,

    /// Flags to pass to the compiler when building for the target
    pub target_rustcflags: Vec<String>,

    /// Whether tests should be optimized by default. Individual test-suites and test files may
    /// override this setting.
    pub optimize_tests: bool,

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
    pub format: OutputFormat,

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

    /// Whether adding git commit information such as the commit hash has been enabled for building
    pub git_hash: bool,

    /// The default Rust edition
    pub edition: Option<String>,

    // Configuration for various run-make tests frobbing things like C compilers
    // or querying about various LLVM component information.
    pub cc: String,
    pub cxx: String,
    pub cflags: String,
    pub cxxflags: String,
    pub ar: String,
    pub target_linker: Option<String>,
    pub host_linker: Option<String>,
    pub llvm_components: String,

    /// Path to a NodeJS executable. Used for JS doctests, emscripten and WASM tests
    pub nodejs: Option<String>,
    /// Path to a npm executable. Used for rustdoc GUI tests
    pub npm: Option<String>,

    /// Whether to rerun tests even if the inputs are unchanged.
    pub force_rerun: bool,

    /// Only rerun the tests that result has been modified accoring to Git status
    pub only_modified: bool,

    pub target_cfgs: AtomicLazyCell<TargetCfgs>,

    pub nocapture: bool,
}

impl Config {
    pub fn run_enabled(&self) -> bool {
        self.run.unwrap_or_else(|| {
            // Auto-detect whether to run based on the platform.
            !self.target.ends_with("-fuchsia")
        })
    }

    pub fn target_cfgs(&self) -> &TargetCfgs {
        match self.target_cfgs.borrow() {
            Some(cfgs) => cfgs,
            None => {
                let _ = self.target_cfgs.fill(TargetCfgs::new(self));
                self.target_cfgs.borrow().unwrap()
            }
        }
    }

    pub fn target_cfg(&self) -> &TargetCfg {
        &self.target_cfgs().current
    }

    pub fn matches_arch(&self, arch: &str) -> bool {
        self.target_cfg().arch == arch ||
        // Shorthand for convenience. The arch for
        // asmjs-unknown-emscripten is actually wasm32.
        (arch == "asmjs" && self.target.starts_with("asmjs")) ||
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

    pub fn has_asm_support(&self) -> bool {
        static ASM_SUPPORTED_ARCHS: &[&str] = &[
            "x86", "x86_64", "arm", "aarch64", "riscv32",
            "riscv64",
            // These targets require an additional asm_experimental_arch feature.
            // "nvptx64", "hexagon", "mips", "mips64", "spirv", "wasm32",
        ];
        ASM_SUPPORTED_ARCHS.contains(&self.target_cfg().arch.as_str())
    }
}

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
}

impl TargetCfgs {
    fn new(config: &Config) -> TargetCfgs {
        let mut targets: HashMap<String, TargetCfg> = serde_json::from_str(&rustc_output(
            config,
            &["--print=all-target-specs-json", "-Zunstable-options"],
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

        // Handle custom target specs, which are not included in `--print=all-target-specs-json`.
        if config.target.ends_with(".json") {
            targets.insert(
                config.target.clone(),
                serde_json::from_str(&rustc_output(
                    config,
                    &["--print=target-spec-json", "-Zunstable-options", "--target", &config.target],
                ))
                .unwrap(),
            );
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
        for config in
            rustc_output(config, &["--print=cfg", "--target", &config.target]).trim().lines()
        {
            let (name, value) = config
                .split_once("=\"")
                .map(|(name, value)| {
                    (
                        name,
                        Some(
                            value
                                .strip_suffix("\"")
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
    #[serde(rename = "target-pointer-width", deserialize_with = "serde_parse_u32")]
    pub(crate) pointer_width: u32,
    #[serde(rename = "target-endian", default)]
    endian: Endian,
    #[serde(rename = "panic-strategy", default)]
    pub(crate) panic: PanicStrategy,
    #[serde(default)]
    pub(crate) dynamic_linking: bool,
}

impl TargetCfg {
    pub(crate) fn os_and_env(&self) -> String {
        format!("{}-{}", self.os, self.env)
    }
}

fn default_os() -> String {
    "none".into()
}

#[derive(Eq, PartialEq, Clone, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Endian {
    #[default]
    Little,
    Big,
}

fn rustc_output(config: &Config, args: &[&str]) -> String {
    let mut command = Command::new(&config.rustc_path);
    add_dylib_path(&mut command, iter::once(&config.compile_lib_path));
    command.args(&config.target_rustcflags).args(args);
    command.env("RUSTC_BOOTSTRAP", "1");

    let output = match command.output() {
        Ok(output) => output,
        Err(e) => panic!("error: failed to run {command:?}: {e}"),
    };
    if !output.status.success() {
        panic!(
            "error: failed to run {command:?}\n--- stdout\n{}\n--- stderr\n{}",
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

fn serde_parse_u32<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let string = String::deserialize(deserializer)?;
    string.parse().map_err(D::Error::custom)
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
    UI_COVERAGE,
];
pub const UI_STDERR: &str = "stderr";
pub const UI_STDOUT: &str = "stdout";
pub const UI_FIXED: &str = "fixed";
pub const UI_RUN_STDERR: &str = "run.stderr";
pub const UI_RUN_STDOUT: &str = "run.stdout";
pub const UI_STDERR_64: &str = "64bit.stderr";
pub const UI_STDERR_32: &str = "32bit.stderr";
pub const UI_STDERR_16: &str = "16bit.stderr";
pub const UI_COVERAGE: &str = "coverage";

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
pub fn incremental_dir(config: &Config, testpaths: &TestPaths, revision: Option<&str>) -> PathBuf {
    output_base_name(config, testpaths, revision).with_extension("inc")
}
