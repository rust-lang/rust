use std::collections::{BTreeSet, HashMap, HashSet};
use std::process::Command;
use std::str::FromStr;
use std::sync::OnceLock;
use std::{fmt, iter};

use build_helper::git::GitConfig;
use camino::{Utf8Path, Utf8PathBuf};
use semver::Version;
use serde::de::{Deserialize, Deserializer, Error as _};

pub use self::Mode::*;
use crate::executor::{ColorConfig, OutputFormat};
use crate::util::{Utf8PathBufExt, add_dylib_path};

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
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($repr => Ok(Self::$variant),)*
                    _ => Err(format!(concat!("unknown `", stringify!($name), "` variant: `{}`"), s)),
                }
            }
        }
    }
}

// Make the macro visible outside of this module, for tests.
#[cfg(test)]
pub(crate) use string_enum;

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Mode {
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

impl Default for Mode {
    fn default() -> Self {
        Mode::Ui
    }
}

impl Mode {
    pub fn aux_dir_disambiguator(self) -> &'static str {
        // Pretty-printing tests could run concurrently, and if they do,
        // they need to keep their output segregated.
        match self {
            Pretty => ".pretty",
            _ => "",
        }
    }

    pub fn output_dir_disambiguator(self) -> &'static str {
        // Coverage tests use the same test files for multiple test modes,
        // so each mode should have a separate output directory.
        match self {
            CoverageMap | CoverageRun => self.to_str(),
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

/// Configuration for compiletest
#[derive(Debug, Default, Clone)]
pub struct Config {
    /// `true` to overwrite stderr/stdout files instead of complaining about changes in output.
    pub bless: bool,

    /// Stop as soon as possible after any test fails.
    /// May run a few more tests before stopping, due to threading.
    pub fail_fast: bool,

    /// The library paths required for running the compiler.
    pub compile_lib_path: Utf8PathBuf,

    /// The library paths required for running compiled programs.
    pub run_lib_path: Utf8PathBuf,

    /// The rustc executable.
    pub rustc_path: Utf8PathBuf,

    /// The cargo executable.
    pub cargo_path: Option<Utf8PathBuf>,

    /// Rustc executable used to compile run-make recipes.
    pub stage0_rustc_path: Option<Utf8PathBuf>,

    /// The rustdoc executable.
    pub rustdoc_path: Option<Utf8PathBuf>,

    /// The coverage-dump executable.
    pub coverage_dump_path: Option<Utf8PathBuf>,

    /// The Python executable to use for LLDB and htmldocck.
    pub python: String,

    /// The jsondocck executable.
    pub jsondocck_path: Option<String>,

    /// The jsondoclint executable.
    pub jsondoclint_path: Option<String>,

    /// The LLVM `FileCheck` binary path.
    pub llvm_filecheck: Option<Utf8PathBuf>,

    /// Path to LLVM's bin directory.
    pub llvm_bin_dir: Option<Utf8PathBuf>,

    /// The path to the Clang executable to run Clang-based tests with. If
    /// `None` then these tests will be ignored.
    pub run_clang_based_tests_with: Option<String>,

    /// The directory containing the sources.
    pub src_root: Utf8PathBuf,
    /// The directory containing the test suite sources. Must be a subdirectory of `src_root`.
    pub src_test_suite_root: Utf8PathBuf,

    /// Root build directory (e.g. `build/`).
    pub build_root: Utf8PathBuf,
    /// Test suite specific build directory (e.g. `build/host/test/ui/`).
    pub build_test_suite_root: Utf8PathBuf,

    /// The directory containing the compiler sysroot
    pub sysroot_base: Utf8PathBuf,

    /// The number of the stage under test.
    pub stage: u32,
    /// The id of the stage under test (stage1-xxx, etc).
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

    /// Whether rustc was built with debug assertions.
    pub with_rustc_debug_assertions: bool,

    /// Whether std was built with debug assertions.
    pub with_std_debug_assertions: bool,

    /// Only run tests that match these filters
    pub filters: Vec<String>,

    /// Skip tests matching these substrings. Corresponds to
    /// `test::TestOpts::skip`. `filter_exact` does not apply to these flags.
    pub skip: Vec<String>,

    /// Exactly match the filter, rather than a substring
    pub filter_exact: bool,

    /// Force the pass mode of a check/build/run-pass test to this mode.
    pub force_pass_mode: Option<PassMode>,

    /// Explicitly enable or disable running.
    pub run: Option<bool>,

    /// A command line to prefix program execution with,
    /// for running under valgrind for example.
    ///
    /// Similar to `CARGO_*_RUNNER` configuration.
    pub runner: Option<String>,

    /// Flags to pass to the compiler when building for the host
    pub host_rustcflags: Vec<String>,

    /// Flags to pass to the compiler when building for the target
    pub target_rustcflags: Vec<String>,

    /// Whether the compiler and stdlib has been built with randomized struct layouts
    pub rust_randomized_layout: bool,

    /// Whether tests should be optimized by default. Individual test-suites and test files may
    /// override this setting.
    pub optimize_tests: bool,

    /// Target system to be tested
    pub target: String,

    /// Host triple for the compiler being invoked
    pub host: String,

    /// Path to / name of the Microsoft Console Debugger (CDB) executable
    pub cdb: Option<Utf8PathBuf>,

    /// Version of CDB
    pub cdb_version: Option<[u16; 4]>,

    /// Path to / name of the GDB executable
    pub gdb: Option<String>,

    /// Version of GDB, encoded as ((major * 1000) + minor) * 1000 + patch
    pub gdb_version: Option<u32>,

    /// Version of LLDB
    pub lldb_version: Option<u32>,

    /// Version of LLVM
    pub llvm_version: Option<Version>,

    /// Is LLVM a system LLVM
    pub system_llvm: bool,

    /// Path to the android tools
    pub android_cross_path: Utf8PathBuf,

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
    pub remote_test_client: Option<Utf8PathBuf>,

    /// mode describing what file the actual ui output will be compared to
    pub compare_mode: Option<CompareMode>,

    /// If true, this will generate a coverage file with UI test files that run `MachineApplicable`
    /// diagnostics but are missing `run-rustfix` annotations. The generated coverage file is
    /// created in `<test_suite_build_root>/rustfix_missing_coverage.txt`
    pub rustfix_coverage: bool,

    /// whether to run `tidy` (html-tidy) when a rustdoc test fails
    pub has_html_tidy: bool,

    /// whether to run `enzyme` autodiff tests
    pub has_enzyme: bool,

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

    /// Only rerun the tests that result has been modified according to Git status
    pub only_modified: bool,

    pub target_cfgs: OnceLock<TargetCfgs>,
    pub builtin_cfg_names: OnceLock<HashSet<String>>,
    pub supported_crate_types: OnceLock<HashSet<String>>,

    pub nocapture: bool,

    // Needed both to construct build_helper::git::GitConfig
    pub nightly_branch: String,
    pub git_merge_commit_email: String,

    /// True if the profiler runtime is enabled for this target.
    /// Used by the "needs-profiler-runtime" directive in test files.
    pub profiler_runtime: bool,

    /// Command for visual diff display, e.g. `diff-tool --color=always`.
    pub diff_command: Option<String>,

    /// Path to minicore aux library, used for `no_core` tests that need `core` stubs in
    /// cross-compilation scenarios that do not otherwise want/need to `-Zbuild-std`. Used in e.g.
    /// ABI tests.
    pub minicore_path: Utf8PathBuf,

    /// If true, disable the "new" executor, and use the older libtest-based
    /// executor to run tests instead. This is a temporary fallback, to make
    /// manual comparative testing easier if bugs are found in the new executor.
    ///
    /// FIXME(Zalathar): Eventually remove this flag and remove the libtest
    /// dependency.
    pub no_new_executor: bool,
}

impl Config {
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
        let mut targets: HashMap<String, TargetCfg> = serde_json::from_str(&rustc_output(
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
                    serde_json::from_str(&rustc_output(
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
        for config in
            rustc_output(config, &["--print=cfg", "--target", &config.target], Default::default())
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
    #[serde(rename = "target-pointer-width", deserialize_with = "serde_parse_u32")]
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
    rustc_output(
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
    let crate_types: HashSet<_> = rustc_output(
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

fn rustc_output(config: &Config, args: &[&str], envs: HashMap<String, String>) -> String {
    let mut command = Command::new(&config.rustc_path);
    add_dylib_path(&mut command, iter::once(&config.compile_lib_path));
    command.args(&config.target_rustcflags).args(args);
    command.env("RUSTC_BOOTSTRAP", "1");
    command.envs(envs);

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
