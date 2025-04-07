//! Serialized configuration of a build.
//!
//! This module implements parsing `bootstrap.toml` configuration files to tweak
//! how the build runs.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{self, Display};
use std::io::IsTerminal;
use std::path::{Path, PathBuf, absolute};
use std::process::Command;
use std::str::FromStr;
use std::sync::OnceLock;
use std::{cmp, env, fs};

use build_helper::ci::CiEnv;
use build_helper::exit;
use build_helper::git::{GitConfig, get_closest_merge_commit, output_result};
use serde::{Deserialize, Deserializer};
use serde_derive::Deserialize;
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

use crate::core::build_steps::compile::CODEGEN_BACKEND_PREFIX;
use crate::core::build_steps::llvm;
use crate::core::build_steps::llvm::LLVM_INVALIDATION_PATHS;
pub use crate::core::config::flags::Subcommand;
use crate::core::config::flags::{Color, Flags, Warnings};
use crate::core::download::is_download_ci_available;
use crate::utils::cache::{INTERNER, Interned};
use crate::utils::channel::{self, GitInfo};
use crate::utils::helpers::{self, exe, output, t};

/// Each path in this list is considered "allowed" in the `download-rustc="if-unchanged"` logic.
/// This means they can be modified and changes to these paths should never trigger a compiler build
/// when "if-unchanged" is set.
///
/// NOTE: Paths must have the ":!" prefix to tell git to ignore changes in those paths during
/// the diff check.
///
/// WARNING: Be cautious when adding paths to this list. If a path that influences the compiler build
/// is added here, it will cause bootstrap to skip necessary rebuilds, which may lead to risky results.
/// For example, "src/bootstrap" should never be included in this list as it plays a crucial role in the
/// final output/compiler, which can be significantly affected by changes made to the bootstrap sources.
#[rustfmt::skip] // We don't want rustfmt to oneline this list
pub(crate) const RUSTC_IF_UNCHANGED_ALLOWED_PATHS: &[&str] = &[
    ":!src/tools",
    ":!src/librustdoc",
    ":!src/rustdoc-json-types",
    ":!tests",
    ":!triagebot.toml",
];

macro_rules! check_ci_llvm {
    ($name:expr) => {
        assert!(
            $name.is_none(),
            "setting {} is incompatible with download-ci-llvm.",
            stringify!($name).replace("_", "-")
        );
    };
}

/// This file is embedded in the overlay directory of the tarball sources. It is
/// useful in scenarios where developers want to see how the tarball sources were
/// generated.
///
/// We also use this file to compare the host's bootstrap.toml against the CI rustc builder
/// configuration to detect any incompatible options.
pub(crate) const BUILDER_CONFIG_FILENAME: &str = "builder-config";

#[derive(Clone, Default)]
pub enum DryRun {
    /// This isn't a dry run.
    #[default]
    Disabled,
    /// This is a dry run enabled by bootstrap itself, so it can verify that no work is done.
    SelfCheck,
    /// This is a dry run enabled by the `--dry-run` flag.
    UserSelected,
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum DebuginfoLevel {
    #[default]
    None,
    LineDirectivesOnly,
    LineTablesOnly,
    Limited,
    Full,
}

// NOTE: can't derive(Deserialize) because the intermediate trip through toml::Value only
// deserializes i64, and derive() only generates visit_u64
impl<'de> Deserialize<'de> for DebuginfoLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        Ok(match Deserialize::deserialize(deserializer)? {
            StringOrInt::String(s) if s == "none" => DebuginfoLevel::None,
            StringOrInt::Int(0) => DebuginfoLevel::None,
            StringOrInt::String(s) if s == "line-directives-only" => {
                DebuginfoLevel::LineDirectivesOnly
            }
            StringOrInt::String(s) if s == "line-tables-only" => DebuginfoLevel::LineTablesOnly,
            StringOrInt::String(s) if s == "limited" => DebuginfoLevel::Limited,
            StringOrInt::Int(1) => DebuginfoLevel::Limited,
            StringOrInt::String(s) if s == "full" => DebuginfoLevel::Full,
            StringOrInt::Int(2) => DebuginfoLevel::Full,
            StringOrInt::Int(n) => {
                let other = serde::de::Unexpected::Signed(n);
                return Err(D::Error::invalid_value(other, &"expected 0, 1, or 2"));
            }
            StringOrInt::String(s) => {
                let other = serde::de::Unexpected::Str(&s);
                return Err(D::Error::invalid_value(
                    other,
                    &"expected none, line-tables-only, limited, or full",
                ));
            }
        })
    }
}

/// Suitable for passing to `-C debuginfo`
impl Display for DebuginfoLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use DebuginfoLevel::*;
        f.write_str(match self {
            None => "0",
            LineDirectivesOnly => "line-directives-only",
            LineTablesOnly => "line-tables-only",
            Limited => "1",
            Full => "2",
        })
    }
}

/// LLD in bootstrap works like this:
/// - Self-contained lld: use `rust-lld` from the compiler's sysroot
/// - External: use an external `lld` binary
///
/// It is configured depending on the target:
/// 1) Everything except MSVC
/// - Self-contained: `-Clinker-flavor=gnu-lld-cc -Clink-self-contained=+linker`
/// - External: `-Clinker-flavor=gnu-lld-cc`
/// 2) MSVC
/// - Self-contained: `-Clinker=<path to rust-lld>`
/// - External: `-Clinker=lld`
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub enum LldMode {
    /// Do not use LLD
    #[default]
    Unused,
    /// Use `rust-lld` from the compiler's sysroot
    SelfContained,
    /// Use an externally provided `lld` binary.
    /// Note that the linker name cannot be overridden, the binary has to be named `lld` and it has
    /// to be in $PATH.
    External,
}

impl LldMode {
    pub fn is_used(&self) -> bool {
        match self {
            LldMode::SelfContained | LldMode::External => true,
            LldMode::Unused => false,
        }
    }
}

/// Determines how will GCC be provided.
#[derive(Default, Clone)]
pub enum GccCiMode {
    /// Build GCC from the local `src/gcc` submodule.
    #[default]
    BuildLocally,
    /// Try to download GCC from CI.
    /// If it is not available on CI, it will be built locally instead.
    DownloadFromCi,
}

/// Global configuration for the entire build and/or bootstrap.
///
/// This structure is parsed from `bootstrap.toml`, and some of the fields are inferred from `git` or build-time parameters.
///
/// Note that this structure is not decoded directly into, but rather it is
/// filled out from the decoded forms of the structs below. For documentation
/// each field, see the corresponding fields in
/// `bootstrap.example.toml`.
#[derive(Default, Clone)]
pub struct Config {
    pub change_id: Option<ChangeId>,
    pub bypass_bootstrap_lock: bool,
    pub ccache: Option<String>,
    /// Call Build::ninja() instead of this.
    pub ninja_in_file: bool,
    pub verbose: usize,
    pub submodules: Option<bool>,
    pub compiler_docs: bool,
    pub library_docs_private_items: bool,
    pub docs_minification: bool,
    pub docs: bool,
    pub locked_deps: bool,
    pub vendor: bool,
    pub target_config: HashMap<TargetSelection, Target>,
    pub full_bootstrap: bool,
    pub bootstrap_cache_path: Option<PathBuf>,
    pub extended: bool,
    pub tools: Option<HashSet<String>>,
    pub sanitizers: bool,
    pub profiler: bool,
    pub omit_git_hash: bool,
    pub skip: Vec<PathBuf>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
    pub test_compare_mode: bool,
    pub color: Color,
    pub patch_binaries_for_nix: Option<bool>,
    pub stage0_metadata: build_helper::stage0_parser::Stage0,
    pub android_ndk: Option<PathBuf>,
    /// Whether to use the `c` feature of the `compiler_builtins` crate.
    pub optimized_compiler_builtins: bool,

    pub stdout_is_tty: bool,
    pub stderr_is_tty: bool,

    pub on_fail: Option<String>,
    pub explicit_stage_from_cli: bool,
    pub explicit_stage_from_config: bool,
    pub stage: u32,
    pub keep_stage: Vec<u32>,
    pub keep_stage_std: Vec<u32>,
    pub src: PathBuf,
    /// defaults to `bootstrap.toml`
    pub config: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub dry_run: DryRun,
    pub dump_bootstrap_shims: bool,
    /// Arguments appearing after `--` to be forwarded to tools,
    /// e.g. `--fix-broken` or test arguments.
    pub free_args: Vec<String>,

    /// `None` if we shouldn't download CI compiler artifacts, or the commit to download if we should.
    #[cfg(not(test))]
    download_rustc_commit: Option<String>,
    #[cfg(test)]
    pub download_rustc_commit: Option<String>,

    pub deny_warnings: bool,
    pub backtrace_on_ice: bool,

    // llvm codegen options
    pub llvm_assertions: bool,
    pub llvm_tests: bool,
    pub llvm_enzyme: bool,
    pub llvm_offload: bool,
    pub llvm_plugins: bool,
    pub llvm_optimize: bool,
    pub llvm_thin_lto: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_static_stdcpp: bool,
    pub llvm_libzstd: bool,
    /// `None` if `llvm_from_ci` is true and we haven't yet downloaded llvm.
    #[cfg(not(test))]
    llvm_link_shared: Cell<Option<bool>>,
    #[cfg(test)]
    pub llvm_link_shared: Cell<Option<bool>>,
    pub llvm_clang_cl: Option<String>,
    pub llvm_targets: Option<String>,
    pub llvm_experimental_targets: Option<String>,
    pub llvm_link_jobs: Option<u32>,
    pub llvm_version_suffix: Option<String>,
    pub llvm_use_linker: Option<String>,
    pub llvm_allow_old_toolchain: bool,
    pub llvm_polly: bool,
    pub llvm_clang: bool,
    pub llvm_enable_warnings: bool,
    pub llvm_from_ci: bool,
    pub llvm_build_config: HashMap<String, String>,

    pub lld_mode: LldMode,
    pub lld_enabled: bool,
    pub llvm_tools_enabled: bool,
    pub llvm_bitcode_linker_enabled: bool,

    pub llvm_cflags: Option<String>,
    pub llvm_cxxflags: Option<String>,
    pub llvm_ldflags: Option<String>,
    pub llvm_use_libcxx: bool,

    // gcc codegen options
    pub gcc_ci_mode: GccCiMode,

    // rust codegen options
    pub rust_optimize: RustOptimize,
    pub rust_codegen_units: Option<u32>,
    pub rust_codegen_units_std: Option<u32>,

    pub rustc_debug_assertions: bool,
    pub std_debug_assertions: bool,

    pub rust_overflow_checks: bool,
    pub rust_overflow_checks_std: bool,
    pub rust_debug_logging: bool,
    pub rust_debuginfo_level_rustc: DebuginfoLevel,
    pub rust_debuginfo_level_std: DebuginfoLevel,
    pub rust_debuginfo_level_tools: DebuginfoLevel,
    pub rust_debuginfo_level_tests: DebuginfoLevel,
    pub rust_rpath: bool,
    pub rust_strip: bool,
    pub rust_frame_pointers: bool,
    pub rust_stack_protector: Option<String>,
    pub rustc_default_linker: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_dist_src: bool,
    pub rust_codegen_backends: Vec<String>,
    pub rust_verify_llvm_ir: bool,
    pub rust_thin_lto_import_instr_limit: Option<u32>,
    pub rust_randomize_layout: bool,
    pub rust_remap_debuginfo: bool,
    pub rust_new_symbol_mangling: Option<bool>,
    pub rust_profile_use: Option<String>,
    pub rust_profile_generate: Option<String>,
    pub rust_lto: RustcLto,
    pub rust_validate_mir_opts: Option<u32>,
    pub rust_std_features: BTreeSet<String>,
    pub llvm_profile_use: Option<String>,
    pub llvm_profile_generate: bool,
    pub llvm_libunwind_default: Option<LlvmLibunwind>,
    pub enable_bolt_settings: bool,

    pub reproducible_artifacts: Vec<String>,

    pub build: TargetSelection,
    pub hosts: Vec<TargetSelection>,
    pub targets: Vec<TargetSelection>,
    pub local_rebuild: bool,
    #[cfg(not(test))]
    jemalloc: bool,
    #[cfg(test)]
    pub jemalloc: bool,
    pub control_flow_guard: bool,
    pub ehcont_guard: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_compression_formats: Option<Vec<String>>,
    pub dist_compression_profile: String,
    pub dist_include_mingw_linker: bool,
    pub dist_vendor: bool,

    // libstd features
    pub backtrace: bool, // support for RUST_BACKTRACE

    // misc
    pub low_priority: bool,
    pub channel: String,
    pub description: Option<String>,
    pub verbose_tests: bool,
    pub save_toolstates: Option<PathBuf>,
    pub print_step_timings: bool,
    pub print_step_rusage: bool,

    // Fallback musl-root for all targets
    pub musl_root: Option<PathBuf>,
    pub prefix: Option<PathBuf>,
    pub sysconfdir: Option<PathBuf>,
    pub datadir: Option<PathBuf>,
    pub docdir: Option<PathBuf>,
    pub bindir: PathBuf,
    pub libdir: Option<PathBuf>,
    pub mandir: Option<PathBuf>,
    pub codegen_tests: bool,
    pub nodejs: Option<PathBuf>,
    pub npm: Option<PathBuf>,
    pub gdb: Option<PathBuf>,
    pub lldb: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub reuse: Option<PathBuf>,
    pub cargo_native_static: bool,
    pub configure_args: Vec<String>,
    pub out: PathBuf,
    pub rust_info: channel::GitInfo,

    pub cargo_info: channel::GitInfo,
    pub rust_analyzer_info: channel::GitInfo,
    pub clippy_info: channel::GitInfo,
    pub miri_info: channel::GitInfo,
    pub rustfmt_info: channel::GitInfo,
    pub enzyme_info: channel::GitInfo,
    pub in_tree_llvm_info: channel::GitInfo,
    pub in_tree_gcc_info: channel::GitInfo,

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,
    pub initial_cargo_clippy: Option<PathBuf>,
    pub initial_sysroot: PathBuf,

    #[cfg(not(test))]
    initial_rustfmt: RefCell<RustfmtState>,
    #[cfg(test)]
    pub initial_rustfmt: RefCell<RustfmtState>,

    /// The paths to work with. For example: with `./x check foo bar` we get
    /// `paths=["foo", "bar"]`.
    pub paths: Vec<PathBuf>,

    /// Command for visual diff display, e.g. `diff-tool --color=always`.
    pub compiletest_diff_tool: Option<String>,

    pub is_running_on_ci: bool,
}

#[derive(Clone, Debug, Default)]
pub enum RustfmtState {
    SystemToolchain(PathBuf),
    Downloaded(PathBuf),
    Unavailable,
    #[default]
    LazyEvaluated,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LlvmLibunwind {
    #[default]
    No,
    InTree,
    System,
}

impl FromStr for LlvmLibunwind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "no" => Ok(Self::No),
            "in-tree" => Ok(Self::InTree),
            "system" => Ok(Self::System),
            invalid => Err(format!("Invalid value '{invalid}' for rust.llvm-libunwind config.")),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitDebuginfo {
    Packed,
    Unpacked,
    #[default]
    Off,
}

impl std::str::FromStr for SplitDebuginfo {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "packed" => Ok(SplitDebuginfo::Packed),
            "unpacked" => Ok(SplitDebuginfo::Unpacked),
            "off" => Ok(SplitDebuginfo::Off),
            _ => Err(()),
        }
    }
}

impl SplitDebuginfo {
    /// Returns the default `-Csplit-debuginfo` value for the current target. See the comment for
    /// `rust.split-debuginfo` in `bootstrap.example.toml`.
    fn default_for_platform(target: TargetSelection) -> Self {
        if target.contains("apple") {
            SplitDebuginfo::Unpacked
        } else if target.is_windows() {
            SplitDebuginfo::Packed
        } else {
            SplitDebuginfo::Off
        }
    }
}

/// LTO mode used for compiling rustc itself.
#[derive(Default, Clone, PartialEq, Debug)]
pub enum RustcLto {
    Off,
    #[default]
    ThinLocal,
    Thin,
    Fat,
}

impl std::str::FromStr for RustcLto {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "thin-local" => Ok(RustcLto::ThinLocal),
            "thin" => Ok(RustcLto::Thin),
            "fat" => Ok(RustcLto::Fat),
            "off" => Ok(RustcLto::Off),
            _ => Err(format!("Invalid value for rustc LTO: {s}")),
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
// N.B.: This type is used everywhere, and the entire codebase relies on it being Copy.
// Making !Copy is highly nontrivial!
pub struct TargetSelection {
    pub triple: Interned<String>,
    file: Option<Interned<String>>,
    synthetic: bool,
}

/// Newtype over `Vec<TargetSelection>` so we can implement custom parsing logic
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TargetSelectionList(Vec<TargetSelection>);

pub fn target_selection_list(s: &str) -> Result<TargetSelectionList, String> {
    Ok(TargetSelectionList(
        s.split(',').filter(|s| !s.is_empty()).map(TargetSelection::from_user).collect(),
    ))
}

impl TargetSelection {
    pub fn from_user(selection: &str) -> Self {
        let path = Path::new(selection);

        let (triple, file) = if path.exists() {
            let triple = path
                .file_stem()
                .expect("Target specification file has no file stem")
                .to_str()
                .expect("Target specification file stem is not UTF-8");

            (triple, Some(selection))
        } else {
            (selection, None)
        };

        let triple = INTERNER.intern_str(triple);
        let file = file.map(|f| INTERNER.intern_str(f));

        Self { triple, file, synthetic: false }
    }

    pub fn create_synthetic(triple: &str, file: &str) -> Self {
        Self {
            triple: INTERNER.intern_str(triple),
            file: Some(INTERNER.intern_str(file)),
            synthetic: true,
        }
    }

    pub fn rustc_target_arg(&self) -> &str {
        self.file.as_ref().unwrap_or(&self.triple)
    }

    pub fn contains(&self, needle: &str) -> bool {
        self.triple.contains(needle)
    }

    pub fn starts_with(&self, needle: &str) -> bool {
        self.triple.starts_with(needle)
    }

    pub fn ends_with(&self, needle: &str) -> bool {
        self.triple.ends_with(needle)
    }

    // See src/bootstrap/synthetic_targets.rs
    pub fn is_synthetic(&self) -> bool {
        self.synthetic
    }

    pub fn is_msvc(&self) -> bool {
        self.contains("msvc")
    }

    pub fn is_windows(&self) -> bool {
        self.contains("windows")
    }

    pub fn is_windows_gnu(&self) -> bool {
        self.ends_with("windows-gnu")
    }

    pub fn is_cygwin(&self) -> bool {
        self.is_windows() &&
        // ref. https://cygwin.com/pipermail/cygwin/2022-February/250802.html
        env::var("OSTYPE").is_ok_and(|v| v.to_lowercase().contains("cygwin"))
    }

    pub fn needs_crt_begin_end(&self) -> bool {
        self.contains("musl") && !self.contains("unikraft")
    }

    /// Path to the file defining the custom target, if any.
    pub fn filepath(&self) -> Option<&Path> {
        self.file.as_ref().map(Path::new)
    }
}

impl fmt::Display for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.triple)?;
        if let Some(file) = self.file {
            write!(f, "({file})")?;
        }
        Ok(())
    }
}

impl fmt::Debug for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl PartialEq<&str> for TargetSelection {
    fn eq(&self, other: &&str) -> bool {
        self.triple == *other
    }
}

// Targets are often used as directory names throughout bootstrap.
// This impl makes it more ergonomics to use them as such.
impl AsRef<Path> for TargetSelection {
    fn as_ref(&self) -> &Path {
        self.triple.as_ref()
    }
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Target {
    /// Some(path to llvm-config) if using an external LLVM.
    pub llvm_config: Option<PathBuf>,
    pub llvm_has_rust_patches: Option<bool>,
    /// Some(path to FileCheck) if one was specified.
    pub llvm_filecheck: Option<PathBuf>,
    pub llvm_libunwind: Option<LlvmLibunwind>,
    pub cc: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub ar: Option<PathBuf>,
    pub ranlib: Option<PathBuf>,
    pub default_linker: Option<PathBuf>,
    pub linker: Option<PathBuf>,
    pub split_debuginfo: Option<SplitDebuginfo>,
    pub sanitizers: Option<bool>,
    pub profiler: Option<StringOrBool>,
    pub rpath: Option<bool>,
    pub crt_static: Option<bool>,
    pub musl_root: Option<PathBuf>,
    pub musl_libdir: Option<PathBuf>,
    pub wasi_root: Option<PathBuf>,
    pub qemu_rootfs: Option<PathBuf>,
    pub runner: Option<String>,
    pub no_std: bool,
    pub codegen_backends: Option<Vec<String>>,
    pub optimized_compiler_builtins: Option<bool>,
    pub jemalloc: Option<bool>,
}

impl Target {
    pub fn from_triple(triple: &str) -> Self {
        let mut target: Self = Default::default();
        if triple.contains("-none") || triple.contains("nvptx") || triple.contains("switch") {
            target.no_std = true;
        }
        if triple.contains("emscripten") {
            target.runner = Some("node".into());
        }
        target
    }
}
/// Structure of the `bootstrap.toml` file that configuration is read from.
///
/// This structure uses `Decodable` to automatically decode a TOML configuration
/// file into this format, and then this is traversed and written into the above
/// `Config` structure.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub(crate) struct TomlConfig {
    #[serde(flatten)]
    change_id: ChangeIdWrapper,
    build: Option<Build>,
    install: Option<Install>,
    llvm: Option<Llvm>,
    gcc: Option<Gcc>,
    rust: Option<Rust>,
    target: Option<HashMap<String, TomlTarget>>,
    dist: Option<Dist>,
    profile: Option<String>,
}

/// This enum is used for deserializing change IDs from TOML, allowing both numeric values and the string `"ignore"`.
#[derive(Clone, Debug, PartialEq)]
pub enum ChangeId {
    Ignore,
    Id(usize),
}

/// Since we use `#[serde(deny_unknown_fields)]` on `TomlConfig`, we need a wrapper type
/// for the "change-id" field to parse it even if other fields are invalid. This ensures
/// that if deserialization fails due to other fields, we can still provide the changelogs
/// to allow developers to potentially find the reason for the failure in the logs..
#[derive(Deserialize, Default)]
pub(crate) struct ChangeIdWrapper {
    #[serde(alias = "change-id", default, deserialize_with = "deserialize_change_id")]
    pub(crate) inner: Option<ChangeId>,
}

fn deserialize_change_id<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ChangeId>, D::Error> {
    let value = toml::Value::deserialize(deserializer)?;
    Ok(match value {
        toml::Value::String(s) if s == "ignore" => Some(ChangeId::Ignore),
        toml::Value::Integer(i) => Some(ChangeId::Id(i as usize)),
        _ => {
            return Err(serde::de::Error::custom(
                "expected \"ignore\" or an integer for change-id",
            ));
        }
    })
}

/// Describes how to handle conflicts in merging two [`TomlConfig`]
#[derive(Copy, Clone, Debug)]
enum ReplaceOpt {
    /// Silently ignore a duplicated value
    IgnoreDuplicate,
    /// Override the current value, even if it's `Some`
    Override,
    /// Exit with an error on duplicate values
    ErrorOnDuplicate,
}

trait Merge {
    fn merge(&mut self, other: Self, replace: ReplaceOpt);
}

impl Merge for TomlConfig {
    fn merge(
        &mut self,
        TomlConfig { build, install, llvm, gcc, rust, dist, target, profile, change_id }: Self,
        replace: ReplaceOpt,
    ) {
        fn do_merge<T: Merge>(x: &mut Option<T>, y: Option<T>, replace: ReplaceOpt) {
            if let Some(new) = y {
                if let Some(original) = x {
                    original.merge(new, replace);
                } else {
                    *x = Some(new);
                }
            }
        }

        self.change_id.inner.merge(change_id.inner, replace);
        self.profile.merge(profile, replace);

        do_merge(&mut self.build, build, replace);
        do_merge(&mut self.install, install, replace);
        do_merge(&mut self.llvm, llvm, replace);
        do_merge(&mut self.gcc, gcc, replace);
        do_merge(&mut self.rust, rust, replace);
        do_merge(&mut self.dist, dist, replace);

        match (self.target.as_mut(), target) {
            (_, None) => {}
            (None, Some(target)) => self.target = Some(target),
            (Some(original_target), Some(new_target)) => {
                for (triple, new) in new_target {
                    if let Some(original) = original_target.get_mut(&triple) {
                        original.merge(new, replace);
                    } else {
                        original_target.insert(triple, new);
                    }
                }
            }
        }
    }
}

// We are using a decl macro instead of a derive proc macro here to reduce the compile time of bootstrap.
macro_rules! define_config {
    ($(#[$attr:meta])* struct $name:ident {
        $($field:ident: Option<$field_ty:ty> = $field_key:literal,)*
    }) => {
        $(#[$attr])*
        struct $name {
            $($field: Option<$field_ty>,)*
        }

        impl Merge for $name {
            fn merge(&mut self, other: Self, replace: ReplaceOpt) {
                $(
                    match replace {
                        ReplaceOpt::IgnoreDuplicate => {
                            if self.$field.is_none() {
                                self.$field = other.$field;
                            }
                        },
                        ReplaceOpt::Override => {
                            if other.$field.is_some() {
                                self.$field = other.$field;
                            }
                        }
                        ReplaceOpt::ErrorOnDuplicate => {
                            if other.$field.is_some() {
                                if self.$field.is_some() {
                                    if cfg!(test) {
                                        panic!("overriding existing option")
                                    } else {
                                        eprintln!("overriding existing option: `{}`", stringify!($field));
                                        exit!(2);
                                    }
                                } else {
                                    self.$field = other.$field;
                                }
                            }
                        }
                    }
                )*
            }
        }

        // The following is a trimmed version of what serde_derive generates. All parts not relevant
        // for toml deserialization have been removed. This reduces the binary size and improves
        // compile time of bootstrap.
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct Field;
                impl<'de> serde::de::Visitor<'de> for Field {
                    type Value = $name;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str(concat!("struct ", stringify!($name)))
                    }

                    #[inline]
                    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::MapAccess<'de>,
                    {
                        $(let mut $field: Option<$field_ty> = None;)*
                        while let Some(key) =
                            match serde::de::MapAccess::next_key::<String>(&mut map) {
                                Ok(val) => val,
                                Err(err) => {
                                    return Err(err);
                                }
                            }
                        {
                            match &*key {
                                $($field_key => {
                                    if $field.is_some() {
                                        return Err(<A::Error as serde::de::Error>::duplicate_field(
                                            $field_key,
                                        ));
                                    }
                                    $field = match serde::de::MapAccess::next_value::<$field_ty>(
                                        &mut map,
                                    ) {
                                        Ok(val) => Some(val),
                                        Err(err) => {
                                            return Err(err);
                                        }
                                    };
                                })*
                                key => {
                                    return Err(serde::de::Error::unknown_field(key, FIELDS));
                                }
                            }
                        }
                        Ok($name { $($field),* })
                    }
                }
                const FIELDS: &'static [&'static str] = &[
                    $($field_key,)*
                ];
                Deserializer::deserialize_struct(
                    deserializer,
                    stringify!($name),
                    FIELDS,
                    Field,
                )
            }
        }
    }
}

impl<T> Merge for Option<T> {
    fn merge(&mut self, other: Self, replace: ReplaceOpt) {
        match replace {
            ReplaceOpt::IgnoreDuplicate => {
                if self.is_none() {
                    *self = other;
                }
            }
            ReplaceOpt::Override => {
                if other.is_some() {
                    *self = other;
                }
            }
            ReplaceOpt::ErrorOnDuplicate => {
                if other.is_some() {
                    if self.is_some() {
                        if cfg!(test) {
                            panic!("overriding existing option")
                        } else {
                            eprintln!("overriding existing option");
                            exit!(2);
                        }
                    } else {
                        *self = other;
                    }
                }
            }
        }
    }
}

define_config! {
    /// TOML representation of various global build decisions.
    #[derive(Default)]
    struct Build {
        build: Option<String> = "build",
        description: Option<String> = "description",
        host: Option<Vec<String>> = "host",
        target: Option<Vec<String>> = "target",
        build_dir: Option<String> = "build-dir",
        cargo: Option<PathBuf> = "cargo",
        rustc: Option<PathBuf> = "rustc",
        rustfmt: Option<PathBuf> = "rustfmt",
        cargo_clippy: Option<PathBuf> = "cargo-clippy",
        docs: Option<bool> = "docs",
        compiler_docs: Option<bool> = "compiler-docs",
        library_docs_private_items: Option<bool> = "library-docs-private-items",
        docs_minification: Option<bool> = "docs-minification",
        submodules: Option<bool> = "submodules",
        gdb: Option<String> = "gdb",
        lldb: Option<String> = "lldb",
        nodejs: Option<String> = "nodejs",
        npm: Option<String> = "npm",
        python: Option<String> = "python",
        reuse: Option<String> = "reuse",
        locked_deps: Option<bool> = "locked-deps",
        vendor: Option<bool> = "vendor",
        full_bootstrap: Option<bool> = "full-bootstrap",
        bootstrap_cache_path: Option<PathBuf> = "bootstrap-cache-path",
        extended: Option<bool> = "extended",
        tools: Option<HashSet<String>> = "tools",
        verbose: Option<usize> = "verbose",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<bool> = "profiler",
        cargo_native_static: Option<bool> = "cargo-native-static",
        low_priority: Option<bool> = "low-priority",
        configure_args: Option<Vec<String>> = "configure-args",
        local_rebuild: Option<bool> = "local-rebuild",
        print_step_timings: Option<bool> = "print-step-timings",
        print_step_rusage: Option<bool> = "print-step-rusage",
        check_stage: Option<u32> = "check-stage",
        doc_stage: Option<u32> = "doc-stage",
        build_stage: Option<u32> = "build-stage",
        test_stage: Option<u32> = "test-stage",
        install_stage: Option<u32> = "install-stage",
        dist_stage: Option<u32> = "dist-stage",
        bench_stage: Option<u32> = "bench-stage",
        patch_binaries_for_nix: Option<bool> = "patch-binaries-for-nix",
        // NOTE: only parsed by bootstrap.py, `--feature build-metrics` enables metrics unconditionally
        metrics: Option<bool> = "metrics",
        android_ndk: Option<PathBuf> = "android-ndk",
        optimized_compiler_builtins: Option<bool> = "optimized-compiler-builtins",
        jobs: Option<u32> = "jobs",
        compiletest_diff_tool: Option<String> = "compiletest-diff-tool",
        ccache: Option<StringOrBool> = "ccache",
        exclude: Option<Vec<PathBuf>> = "exclude",
    }
}

define_config! {
    /// TOML representation of various global install decisions.
    struct Install {
        prefix: Option<String> = "prefix",
        sysconfdir: Option<String> = "sysconfdir",
        docdir: Option<String> = "docdir",
        bindir: Option<String> = "bindir",
        libdir: Option<String> = "libdir",
        mandir: Option<String> = "mandir",
        datadir: Option<String> = "datadir",
    }
}

define_config! {
    /// TOML representation of how the LLVM build is configured.
    struct Llvm {
        optimize: Option<bool> = "optimize",
        thin_lto: Option<bool> = "thin-lto",
        release_debuginfo: Option<bool> = "release-debuginfo",
        assertions: Option<bool> = "assertions",
        tests: Option<bool> = "tests",
        enzyme: Option<bool> = "enzyme",
        plugins: Option<bool> = "plugins",
        // FIXME: Remove this field at Q2 2025, it has been replaced by build.ccache
        ccache: Option<StringOrBool> = "ccache",
        static_libstdcpp: Option<bool> = "static-libstdcpp",
        libzstd: Option<bool> = "libzstd",
        ninja: Option<bool> = "ninja",
        targets: Option<String> = "targets",
        experimental_targets: Option<String> = "experimental-targets",
        link_jobs: Option<u32> = "link-jobs",
        link_shared: Option<bool> = "link-shared",
        version_suffix: Option<String> = "version-suffix",
        clang_cl: Option<String> = "clang-cl",
        cflags: Option<String> = "cflags",
        cxxflags: Option<String> = "cxxflags",
        ldflags: Option<String> = "ldflags",
        use_libcxx: Option<bool> = "use-libcxx",
        use_linker: Option<String> = "use-linker",
        allow_old_toolchain: Option<bool> = "allow-old-toolchain",
        offload: Option<bool> = "offload",
        polly: Option<bool> = "polly",
        clang: Option<bool> = "clang",
        enable_warnings: Option<bool> = "enable-warnings",
        download_ci_llvm: Option<StringOrBool> = "download-ci-llvm",
        build_config: Option<HashMap<String, String>> = "build-config",
    }
}

define_config! {
    /// TOML representation of how the GCC build is configured.
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
    }
}

define_config! {
    struct Dist {
        sign_folder: Option<String> = "sign-folder",
        upload_addr: Option<String> = "upload-addr",
        src_tarball: Option<bool> = "src-tarball",
        compression_formats: Option<Vec<String>> = "compression-formats",
        compression_profile: Option<String> = "compression-profile",
        include_mingw_linker: Option<bool> = "include-mingw-linker",
        vendor: Option<bool> = "vendor",
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum StringOrBool {
    String(String),
    Bool(bool),
}

impl Default for StringOrBool {
    fn default() -> StringOrBool {
        StringOrBool::Bool(false)
    }
}

impl StringOrBool {
    fn is_string_or_true(&self) -> bool {
        matches!(self, Self::String(_) | Self::Bool(true))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustOptimize {
    String(String),
    Int(u8),
    Bool(bool),
}

impl Default for RustOptimize {
    fn default() -> RustOptimize {
        RustOptimize::Bool(false)
    }
}

impl<'de> Deserialize<'de> for RustOptimize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OptimizeVisitor)
    }
}

struct OptimizeVisitor;

impl serde::de::Visitor<'_> for OptimizeVisitor {
    type Value = RustOptimize;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(r#"one of: 0, 1, 2, 3, "s", "z", true, false"#)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if matches!(value, "s" | "z") {
            Ok(RustOptimize::String(value.to_string()))
        } else {
            Err(serde::de::Error::custom(format_optimize_error_msg(value)))
        }
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if matches!(value, 0..=3) {
            Ok(RustOptimize::Int(value as u8))
        } else {
            Err(serde::de::Error::custom(format_optimize_error_msg(value)))
        }
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(RustOptimize::Bool(value))
    }
}

fn format_optimize_error_msg(v: impl std::fmt::Display) -> String {
    format!(
        r#"unrecognized option for rust optimize: "{v}", expected one of 0, 1, 2, 3, "s", "z", true, false"#
    )
}

impl RustOptimize {
    pub(crate) fn is_release(&self) -> bool {
        match &self {
            RustOptimize::Bool(true) | RustOptimize::String(_) => true,
            RustOptimize::Int(i) => *i > 0,
            RustOptimize::Bool(false) => false,
        }
    }

    pub(crate) fn get_opt_level(&self) -> Option<String> {
        match &self {
            RustOptimize::String(s) => Some(s.clone()),
            RustOptimize::Int(i) => Some(i.to_string()),
            RustOptimize::Bool(_) => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrInt {
    String(String),
    Int(i64),
}

impl<'de> Deserialize<'de> for LldMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LldModeVisitor;

        impl serde::de::Visitor<'_> for LldModeVisitor {
            type Value = LldMode;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("one of true, 'self-contained' or 'external'")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(if v { LldMode::External } else { LldMode::Unused })
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match v {
                    "external" => Ok(LldMode::External),
                    "self-contained" => Ok(LldMode::SelfContained),
                    _ => Err(E::custom(format!("unknown mode {v}"))),
                }
            }
        }

        deserializer.deserialize_any(LldModeVisitor)
    }
}

define_config! {
    /// TOML representation of how the Rust build is configured.
    struct Rust {
        optimize: Option<RustOptimize> = "optimize",
        debug: Option<bool> = "debug",
        codegen_units: Option<u32> = "codegen-units",
        codegen_units_std: Option<u32> = "codegen-units-std",
        rustc_debug_assertions: Option<bool> = "debug-assertions",
        randomize_layout: Option<bool> = "randomize-layout",
        std_debug_assertions: Option<bool> = "debug-assertions-std",
        overflow_checks: Option<bool> = "overflow-checks",
        overflow_checks_std: Option<bool> = "overflow-checks-std",
        debug_logging: Option<bool> = "debug-logging",
        debuginfo_level: Option<DebuginfoLevel> = "debuginfo-level",
        debuginfo_level_rustc: Option<DebuginfoLevel> = "debuginfo-level-rustc",
        debuginfo_level_std: Option<DebuginfoLevel> = "debuginfo-level-std",
        debuginfo_level_tools: Option<DebuginfoLevel> = "debuginfo-level-tools",
        debuginfo_level_tests: Option<DebuginfoLevel> = "debuginfo-level-tests",
        backtrace: Option<bool> = "backtrace",
        incremental: Option<bool> = "incremental",
        default_linker: Option<String> = "default-linker",
        channel: Option<String> = "channel",
        // FIXME: Remove this field at Q2 2025, it has been replaced by build.description
        description: Option<String> = "description",
        musl_root: Option<String> = "musl-root",
        rpath: Option<bool> = "rpath",
        strip: Option<bool> = "strip",
        frame_pointers: Option<bool> = "frame-pointers",
        stack_protector: Option<String> = "stack-protector",
        verbose_tests: Option<bool> = "verbose-tests",
        optimize_tests: Option<bool> = "optimize-tests",
        codegen_tests: Option<bool> = "codegen-tests",
        omit_git_hash: Option<bool> = "omit-git-hash",
        dist_src: Option<bool> = "dist-src",
        save_toolstates: Option<String> = "save-toolstates",
        codegen_backends: Option<Vec<String>> = "codegen-backends",
        llvm_bitcode_linker: Option<bool> = "llvm-bitcode-linker",
        lld: Option<bool> = "lld",
        lld_mode: Option<LldMode> = "use-lld",
        llvm_tools: Option<bool> = "llvm-tools",
        deny_warnings: Option<bool> = "deny-warnings",
        backtrace_on_ice: Option<bool> = "backtrace-on-ice",
        verify_llvm_ir: Option<bool> = "verify-llvm-ir",
        thin_lto_import_instr_limit: Option<u32> = "thin-lto-import-instr-limit",
        remap_debuginfo: Option<bool> = "remap-debuginfo",
        jemalloc: Option<bool> = "jemalloc",
        test_compare_mode: Option<bool> = "test-compare-mode",
        llvm_libunwind: Option<String> = "llvm-libunwind",
        control_flow_guard: Option<bool> = "control-flow-guard",
        ehcont_guard: Option<bool> = "ehcont-guard",
        new_symbol_mangling: Option<bool> = "new-symbol-mangling",
        profile_generate: Option<String> = "profile-generate",
        profile_use: Option<String> = "profile-use",
        // ignored; this is set from an env var set by bootstrap.py
        download_rustc: Option<StringOrBool> = "download-rustc",
        lto: Option<String> = "lto",
        validate_mir_opts: Option<u32> = "validate-mir-opts",
        std_features: Option<BTreeSet<String>> = "std-features",
    }
}

define_config! {
    /// TOML representation of how each build target is configured.
    struct TomlTarget {
        cc: Option<String> = "cc",
        cxx: Option<String> = "cxx",
        ar: Option<String> = "ar",
        ranlib: Option<String> = "ranlib",
        default_linker: Option<PathBuf> = "default-linker",
        linker: Option<String> = "linker",
        split_debuginfo: Option<String> = "split-debuginfo",
        llvm_config: Option<String> = "llvm-config",
        llvm_has_rust_patches: Option<bool> = "llvm-has-rust-patches",
        llvm_filecheck: Option<String> = "llvm-filecheck",
        llvm_libunwind: Option<String> = "llvm-libunwind",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<StringOrBool> = "profiler",
        rpath: Option<bool> = "rpath",
        crt_static: Option<bool> = "crt-static",
        musl_root: Option<String> = "musl-root",
        musl_libdir: Option<String> = "musl-libdir",
        wasi_root: Option<String> = "wasi-root",
        qemu_rootfs: Option<String> = "qemu-rootfs",
        no_std: Option<bool> = "no-std",
        codegen_backends: Option<Vec<String>> = "codegen-backends",
        runner: Option<String> = "runner",
        optimized_compiler_builtins: Option<bool> = "optimized-compiler-builtins",
        jemalloc: Option<bool> = "jemalloc",
    }
}

impl Config {
    #[cfg_attr(
        feature = "tracing",
        instrument(target = "CONFIG_HANDLING", level = "trace", name = "Config::default_opts")
    )]
    pub fn default_opts() -> Config {
        #[cfg(feature = "tracing")]
        span!(target: "CONFIG_HANDLING", tracing::Level::TRACE, "constructing default config");

        Config {
            bypass_bootstrap_lock: false,
            llvm_optimize: true,
            ninja_in_file: true,
            llvm_static_stdcpp: false,
            llvm_libzstd: false,
            backtrace: true,
            rust_optimize: RustOptimize::Bool(true),
            rust_optimize_tests: true,
            rust_randomize_layout: false,
            submodules: None,
            docs: true,
            docs_minification: true,
            rust_rpath: true,
            rust_strip: false,
            channel: "dev".to_string(),
            codegen_tests: true,
            rust_dist_src: true,
            rust_codegen_backends: vec!["llvm".to_owned()],
            deny_warnings: true,
            bindir: "bin".into(),
            dist_include_mingw_linker: true,
            dist_compression_profile: "fast".into(),

            stdout_is_tty: std::io::stdout().is_terminal(),
            stderr_is_tty: std::io::stderr().is_terminal(),

            // set by build.rs
            build: TargetSelection::from_user(env!("BUILD_TRIPLE")),

            src: {
                let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                // Undo `src/bootstrap`
                manifest_dir.parent().unwrap().parent().unwrap().to_owned()
            },
            out: PathBuf::from("build"),

            // This is needed by codegen_ssa on macOS to ship `llvm-objcopy` aliased to
            // `rust-objcopy` to workaround bad `strip`s on macOS.
            llvm_tools_enabled: true,

            ..Default::default()
        }
    }

    pub(crate) fn get_builder_toml(&self, build_name: &str) -> Result<TomlConfig, toml::de::Error> {
        if self.dry_run() {
            return Ok(TomlConfig::default());
        }

        let builder_config_path =
            self.out.join(self.build.triple).join(build_name).join(BUILDER_CONFIG_FILENAME);
        Self::get_toml(&builder_config_path)
    }

    #[cfg(test)]
    pub(crate) fn get_toml(_: &Path) -> Result<TomlConfig, toml::de::Error> {
        Ok(TomlConfig::default())
    }

    #[cfg(not(test))]
    pub(crate) fn get_toml(file: &Path) -> Result<TomlConfig, toml::de::Error> {
        let contents =
            t!(fs::read_to_string(file), format!("config file {} not found", file.display()));
        // Deserialize to Value and then TomlConfig to prevent the Deserialize impl of
        // TomlConfig and sub types to be monomorphized 5x by toml.
        toml::from_str(&contents)
            .and_then(|table: toml::Value| TomlConfig::deserialize(table))
            .inspect_err(|_| {
                if let Ok(ChangeIdWrapper { inner: Some(ChangeId::Id(id)) }) =
                    toml::from_str::<toml::Value>(&contents)
                        .and_then(|table: toml::Value| ChangeIdWrapper::deserialize(table))
                {
                    let changes = crate::find_recent_config_change_ids(id);
                    if !changes.is_empty() {
                        println!(
                            "WARNING: There have been changes to x.py since you last updated:\n{}",
                            crate::human_readable_changes(changes)
                        );
                    }
                }
            })
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(target = "CONFIG_HANDLING", level = "trace", name = "Config::parse", skip_all)
    )]
    pub fn parse(flags: Flags) -> Config {
        Self::parse_inner(flags, Self::get_toml)
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            target = "CONFIG_HANDLING",
            level = "trace",
            name = "Config::parse_inner",
            skip_all
        )
    )]
    pub(crate) fn parse_inner(
        mut flags: Flags,
        get_toml: impl Fn(&Path) -> Result<TomlConfig, toml::de::Error>,
    ) -> Config {
        let mut config = Config::default_opts();

        // Set flags.
        config.paths = std::mem::take(&mut flags.paths);

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "collecting paths and path exclusions",
            "flags.paths" = ?flags.paths,
            "flags.skip" = ?flags.skip,
            "flags.exclude" = ?flags.exclude
        );

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "normalizing and combining `flag.skip`/`flag.exclude` paths",
            "config.skip" = ?config.skip,
        );

        config.include_default_paths = flags.include_default_paths;
        config.rustc_error_format = flags.rustc_error_format;
        config.json_output = flags.json_output;
        config.on_fail = flags.on_fail;
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.dry_run = if flags.dry_run { DryRun::UserSelected } else { DryRun::Disabled };
        config.dump_bootstrap_shims = flags.dump_bootstrap_shims;
        config.keep_stage = flags.keep_stage;
        config.keep_stage_std = flags.keep_stage_std;
        config.color = flags.color;
        config.free_args = std::mem::take(&mut flags.free_args);
        config.llvm_profile_use = flags.llvm_profile_use;
        config.llvm_profile_generate = flags.llvm_profile_generate;
        config.enable_bolt_settings = flags.enable_bolt_settings;
        config.bypass_bootstrap_lock = flags.bypass_bootstrap_lock;
        config.is_running_on_ci = flags.ci.unwrap_or(CiEnv::is_ci());

        // Infer the rest of the configuration.

        if let Some(src) = flags.src {
            config.src = src
        } else {
            // Infer the source directory. This is non-trivial because we want to support a downloaded bootstrap binary,
            // running on a completely different machine from where it was compiled.
            let mut cmd = helpers::git(None);
            // NOTE: we cannot support running from outside the repository because the only other path we have available
            // is set at compile time, which can be wrong if bootstrap was downloaded rather than compiled locally.
            // We still support running outside the repository if we find we aren't in a git directory.

            // NOTE: We get a relative path from git to work around an issue on MSYS/mingw. If we used an absolute path,
            // and end up using MSYS's git rather than git-for-windows, we would get a unix-y MSYS path. But as bootstrap
            // has already been (kinda-cross-)compiled to Windows land, we require a normal Windows path.
            cmd.arg("rev-parse").arg("--show-cdup");
            // Discard stderr because we expect this to fail when building from a tarball.
            let output = cmd
                .as_command_mut()
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
                .and_then(|output| if output.status.success() { Some(output) } else { None });
            if let Some(output) = output {
                let git_root_relative = String::from_utf8(output.stdout).unwrap();
                // We need to canonicalize this path to make sure it uses backslashes instead of forward slashes,
                // and to resolve any relative components.
                let git_root = env::current_dir()
                    .unwrap()
                    .join(PathBuf::from(git_root_relative.trim()))
                    .canonicalize()
                    .unwrap();
                let s = git_root.to_str().unwrap();

                // Bootstrap is quite bad at handling /? in front of paths
                let git_root = match s.strip_prefix("\\\\?\\") {
                    Some(p) => PathBuf::from(p),
                    None => git_root,
                };
                // If this doesn't have at least `stage0`, we guessed wrong. This can happen when,
                // for example, the build directory is inside of another unrelated git directory.
                // In that case keep the original `CARGO_MANIFEST_DIR` handling.
                //
                // NOTE: this implies that downloadable bootstrap isn't supported when the build directory is outside
                // the source directory. We could fix that by setting a variable from all three of python, ./x, and x.ps1.
                if git_root.join("src").join("stage0").exists() {
                    config.src = git_root;
                }
            } else {
                // We're building from a tarball, not git sources.
                // We don't support pre-downloaded bootstrap in this case.
            }
        }

        if cfg!(test) {
            // Use the build directory of the original x.py invocation, so that we can set `initial_rustc` properly.
            config.out = Path::new(
                &env::var_os("CARGO_TARGET_DIR").expect("cargo test directly is not supported"),
            )
            .parent()
            .unwrap()
            .to_path_buf();
        }

        config.stage0_metadata = build_helper::stage0_parser::parse_stage0_file();

        // Locate the configuration file using the following priority (first match wins):
        // 1. `--config <path>` (explicit flag)
        // 2. `RUST_BOOTSTRAP_CONFIG` environment variable
        // 3. `./bootstrap.toml` (local file)
        // 4. `<root>/bootstrap.toml`
        // 5. `./config.toml` (fallback for backward compatibility)
        // 6. `<root>/config.toml`
        let toml_path = flags
            .config
            .clone()
            .or_else(|| env::var_os("RUST_BOOTSTRAP_CONFIG").map(PathBuf::from));
        let using_default_path = toml_path.is_none();
        let mut toml_path = toml_path.unwrap_or_else(|| PathBuf::from("bootstrap.toml"));

        if using_default_path && !toml_path.exists() {
            toml_path = config.src.join(PathBuf::from("bootstrap.toml"));
            if !toml_path.exists() {
                toml_path = PathBuf::from("config.toml");
                if !toml_path.exists() {
                    toml_path = config.src.join(PathBuf::from("config.toml"));
                }
            }
        }

        let file_content = t!(fs::read_to_string(config.src.join("src/ci/channel")));
        let ci_channel = file_content.trim_end();

        // Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
        // but not if `bootstrap.toml` hasn't been created.
        let mut toml = if !using_default_path || toml_path.exists() {
            config.config = Some(if cfg!(not(test)) {
                toml_path.canonicalize().unwrap()
            } else {
                toml_path.clone()
            });
            get_toml(&toml_path).unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to parse '{}': {e}", toml_path.display());
                exit!(2);
            })
        } else {
            config.config = None;
            TomlConfig::default()
        };

        if cfg!(test) {
            // When configuring bootstrap for tests, make sure to set the rustc and Cargo to the
            // same ones used to call the tests (if custom ones are not defined in the toml). If we
            // don't do that, bootstrap will use its own detection logic to find a suitable rustc
            // and Cargo, which doesn't work when the caller is specìfying a custom local rustc or
            // Cargo in their bootstrap.toml.
            let build = toml.build.get_or_insert_with(Default::default);
            build.rustc = build.rustc.take().or(std::env::var_os("RUSTC").map(|p| p.into()));
            build.cargo = build.cargo.take().or(std::env::var_os("CARGO").map(|p| p.into()));
        }

        if GitInfo::new(false, &config.src).is_from_tarball() && toml.profile.is_none() {
            toml.profile = Some("dist".into());
        }

        if let Some(include) = &toml.profile {
            // Allows creating alias for profile names, allowing
            // profiles to be renamed while maintaining back compatibility
            // Keep in sync with `profile_aliases` in bootstrap.py
            let profile_aliases = HashMap::from([("user", "dist")]);
            let include = match profile_aliases.get(include.as_str()) {
                Some(alias) => alias,
                None => include.as_str(),
            };
            let mut include_path = config.src.clone();
            include_path.push("src");
            include_path.push("bootstrap");
            include_path.push("defaults");
            include_path.push(format!("bootstrap.{include}.toml"));
            let included_toml = get_toml(&include_path).unwrap_or_else(|e| {
                eprintln!(
                    "ERROR: Failed to parse default config profile at '{}': {e}",
                    include_path.display()
                );
                exit!(2);
            });
            toml.merge(included_toml, ReplaceOpt::IgnoreDuplicate);
        }

        let mut override_toml = TomlConfig::default();
        for option in flags.set.iter() {
            fn get_table(option: &str) -> Result<TomlConfig, toml::de::Error> {
                toml::from_str(option).and_then(|table: toml::Value| TomlConfig::deserialize(table))
            }

            let mut err = match get_table(option) {
                Ok(v) => {
                    override_toml.merge(v, ReplaceOpt::ErrorOnDuplicate);
                    continue;
                }
                Err(e) => e,
            };
            // We want to be able to set string values without quotes,
            // like in `configure.py`. Try adding quotes around the right hand side
            if let Some((key, value)) = option.split_once('=') {
                if !value.contains('"') {
                    match get_table(&format!(r#"{key}="{value}""#)) {
                        Ok(v) => {
                            override_toml.merge(v, ReplaceOpt::ErrorOnDuplicate);
                            continue;
                        }
                        Err(e) => err = e,
                    }
                }
            }
            eprintln!("failed to parse override `{option}`: `{err}");
            exit!(2)
        }
        toml.merge(override_toml, ReplaceOpt::Override);

        config.change_id = toml.change_id.inner;

        let Build {
            mut description,
            build,
            host,
            target,
            build_dir,
            cargo,
            rustc,
            rustfmt,
            cargo_clippy,
            docs,
            compiler_docs,
            library_docs_private_items,
            docs_minification,
            submodules,
            gdb,
            lldb,
            nodejs,
            npm,
            python,
            reuse,
            locked_deps,
            vendor,
            full_bootstrap,
            bootstrap_cache_path,
            extended,
            tools,
            verbose,
            sanitizers,
            profiler,
            cargo_native_static,
            low_priority,
            configure_args,
            local_rebuild,
            print_step_timings,
            print_step_rusage,
            check_stage,
            doc_stage,
            build_stage,
            test_stage,
            install_stage,
            dist_stage,
            bench_stage,
            patch_binaries_for_nix,
            // This field is only used by bootstrap.py
            metrics: _,
            android_ndk,
            optimized_compiler_builtins,
            jobs,
            compiletest_diff_tool,
            mut ccache,
            exclude,
        } = toml.build.unwrap_or_default();

        let mut paths: Vec<PathBuf> = flags.skip.into_iter().chain(flags.exclude).collect();

        if let Some(exclude) = exclude {
            paths.extend(exclude);
        }

        config.skip = paths
            .into_iter()
            .map(|p| {
                // Never return top-level path here as it would break `--skip`
                // logic on rustc's internal test framework which is utilized
                // by compiletest.
                if cfg!(windows) {
                    PathBuf::from(p.to_str().unwrap().replace('/', "\\"))
                } else {
                    p
                }
            })
            .collect();

        config.jobs = Some(threads_from_config(flags.jobs.unwrap_or(jobs.unwrap_or(0))));

        if let Some(file_build) = build {
            config.build = TargetSelection::from_user(&file_build);
        };

        set(&mut config.out, flags.build_dir.or_else(|| build_dir.map(PathBuf::from)));
        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        if !config.out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            config.out = absolute(&config.out).expect("can't make empty path absolute");
        }

        if cargo_clippy.is_some() && rustc.is_none() {
            println!(
                "WARNING: Using `build.cargo-clippy` without `build.rustc` usually fails due to toolchain conflict."
            );
        }

        config.initial_rustc = if let Some(rustc) = rustc {
            if !flags.skip_stage0_validation {
                config.check_stage0_version(&rustc, "rustc");
            }
            rustc
        } else {
            config.download_beta_toolchain();
            config
                .out
                .join(config.build)
                .join("stage0")
                .join("bin")
                .join(exe("rustc", config.build))
        };

        config.initial_sysroot = config.initial_rustc.ancestors().nth(2).unwrap().into();

        config.initial_cargo_clippy = cargo_clippy;

        config.initial_cargo = if let Some(cargo) = cargo {
            if !flags.skip_stage0_validation {
                config.check_stage0_version(&cargo, "cargo");
            }
            cargo
        } else {
            config.download_beta_toolchain();
            config.initial_sysroot.join("bin").join(exe("cargo", config.build))
        };

        // NOTE: it's important this comes *after* we set `initial_rustc` just above.
        if config.dry_run() {
            let dir = config.out.join("tmp-dry-run");
            t!(fs::create_dir_all(&dir));
            config.out = dir;
        }

        config.hosts = if let Some(TargetSelectionList(arg_host)) = flags.host {
            arg_host
        } else if let Some(file_host) = host {
            file_host.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            vec![config.build]
        };
        config.targets = if let Some(TargetSelectionList(arg_target)) = flags.target {
            arg_target
        } else if let Some(file_target) = target {
            file_target.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            // If target is *not* configured, then default to the host
            // toolchains.
            config.hosts.clone()
        };

        config.nodejs = nodejs.map(PathBuf::from);
        config.npm = npm.map(PathBuf::from);
        config.gdb = gdb.map(PathBuf::from);
        config.lldb = lldb.map(PathBuf::from);
        config.python = python.map(PathBuf::from);
        config.reuse = reuse.map(PathBuf::from);
        config.submodules = submodules;
        config.android_ndk = android_ndk;
        config.bootstrap_cache_path = bootstrap_cache_path;
        set(&mut config.low_priority, low_priority);
        set(&mut config.compiler_docs, compiler_docs);
        set(&mut config.library_docs_private_items, library_docs_private_items);
        set(&mut config.docs_minification, docs_minification);
        set(&mut config.docs, docs);
        set(&mut config.locked_deps, locked_deps);
        set(&mut config.full_bootstrap, full_bootstrap);
        set(&mut config.extended, extended);
        config.tools = tools;
        set(&mut config.verbose, verbose);
        set(&mut config.sanitizers, sanitizers);
        set(&mut config.profiler, profiler);
        set(&mut config.cargo_native_static, cargo_native_static);
        set(&mut config.configure_args, configure_args);
        set(&mut config.local_rebuild, local_rebuild);
        set(&mut config.print_step_timings, print_step_timings);
        set(&mut config.print_step_rusage, print_step_rusage);
        config.patch_binaries_for_nix = patch_binaries_for_nix;

        config.verbose = cmp::max(config.verbose, flags.verbose as usize);

        // Verbose flag is a good default for `rust.verbose-tests`.
        config.verbose_tests = config.is_verbose();

        if let Some(install) = toml.install {
            let Install { prefix, sysconfdir, docdir, bindir, libdir, mandir, datadir } = install;
            config.prefix = prefix.map(PathBuf::from);
            config.sysconfdir = sysconfdir.map(PathBuf::from);
            config.datadir = datadir.map(PathBuf::from);
            config.docdir = docdir.map(PathBuf::from);
            set(&mut config.bindir, bindir.map(PathBuf::from));
            config.libdir = libdir.map(PathBuf::from);
            config.mandir = mandir.map(PathBuf::from);
        }

        config.llvm_assertions =
            toml.llvm.as_ref().is_some_and(|llvm| llvm.assertions.unwrap_or(false));

        // Store off these values as options because if they're not provided
        // we'll infer default values for them later
        let mut llvm_tests = None;
        let mut llvm_enzyme = None;
        let mut llvm_offload = None;
        let mut llvm_plugins = None;
        let mut debug = None;
        let mut rustc_debug_assertions = None;
        let mut std_debug_assertions = None;
        let mut overflow_checks = None;
        let mut overflow_checks_std = None;
        let mut debug_logging = None;
        let mut debuginfo_level = None;
        let mut debuginfo_level_rustc = None;
        let mut debuginfo_level_std = None;
        let mut debuginfo_level_tools = None;
        let mut debuginfo_level_tests = None;
        let mut optimize = None;
        let mut lld_enabled = None;
        let mut std_features = None;

        let is_user_configured_rust_channel =
            if let Some(channel) = toml.rust.as_ref().and_then(|r| r.channel.clone()) {
                if channel == "auto-detect" {
                    config.channel = ci_channel.into();
                } else {
                    config.channel = channel;
                }
                true
            } else {
                false
            };

        let default = config.channel == "dev";
        config.omit_git_hash = toml.rust.as_ref().and_then(|r| r.omit_git_hash).unwrap_or(default);

        config.rust_info = GitInfo::new(config.omit_git_hash, &config.src);
        config.cargo_info = GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/cargo"));
        config.rust_analyzer_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/rust-analyzer"));
        config.clippy_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/clippy"));
        config.miri_info = GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/miri"));
        config.rustfmt_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/rustfmt"));
        config.enzyme_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/enzyme"));
        config.in_tree_llvm_info = GitInfo::new(false, &config.src.join("src/llvm-project"));
        config.in_tree_gcc_info = GitInfo::new(false, &config.src.join("src/gcc"));

        config.vendor = vendor.unwrap_or(
            config.rust_info.is_from_tarball()
                && config.src.join("vendor").exists()
                && config.src.join(".cargo/config.toml").exists(),
        );

        if let Some(rust) = toml.rust {
            let Rust {
                optimize: optimize_toml,
                debug: debug_toml,
                codegen_units,
                codegen_units_std,
                rustc_debug_assertions: rustc_debug_assertions_toml,
                std_debug_assertions: std_debug_assertions_toml,
                overflow_checks: overflow_checks_toml,
                overflow_checks_std: overflow_checks_std_toml,
                debug_logging: debug_logging_toml,
                debuginfo_level: debuginfo_level_toml,
                debuginfo_level_rustc: debuginfo_level_rustc_toml,
                debuginfo_level_std: debuginfo_level_std_toml,
                debuginfo_level_tools: debuginfo_level_tools_toml,
                debuginfo_level_tests: debuginfo_level_tests_toml,
                backtrace,
                incremental,
                randomize_layout,
                default_linker,
                channel: _, // already handled above
                description: rust_description,
                musl_root,
                rpath,
                verbose_tests,
                optimize_tests,
                codegen_tests,
                omit_git_hash: _, // already handled above
                dist_src,
                save_toolstates,
                codegen_backends,
                lld: lld_enabled_toml,
                llvm_tools,
                llvm_bitcode_linker,
                deny_warnings,
                backtrace_on_ice,
                verify_llvm_ir,
                thin_lto_import_instr_limit,
                remap_debuginfo,
                jemalloc,
                test_compare_mode,
                llvm_libunwind,
                control_flow_guard,
                ehcont_guard,
                new_symbol_mangling,
                profile_generate,
                profile_use,
                download_rustc,
                lto,
                validate_mir_opts,
                frame_pointers,
                stack_protector,
                strip,
                lld_mode,
                std_features: std_features_toml,
            } = rust;

            // FIXME(#133381): alt rustc builds currently do *not* have rustc debug assertions
            // enabled. We should not download a CI alt rustc if we need rustc to have debug
            // assertions (e.g. for crashes test suite). This can be changed once something like
            // [Enable debug assertions on alt
            // builds](https://github.com/rust-lang/rust/pull/131077) lands.
            //
            // Note that `rust.debug = true` currently implies `rust.debug-assertions = true`!
            //
            // This relies also on the fact that the global default for `download-rustc` will be
            // `false` if it's not explicitly set.
            let debug_assertions_requested = matches!(rustc_debug_assertions_toml, Some(true))
                || (matches!(debug_toml, Some(true))
                    && !matches!(rustc_debug_assertions_toml, Some(false)));

            if debug_assertions_requested {
                if let Some(ref opt) = download_rustc {
                    if opt.is_string_or_true() {
                        eprintln!(
                            "WARN: currently no CI rustc builds have rustc debug assertions \
                            enabled. Please either set `rust.debug-assertions` to `false` if you \
                            want to use download CI rustc or set `rust.download-rustc` to `false`."
                        );
                    }
                }
            }

            config.download_rustc_commit = config.download_ci_rustc_commit(
                download_rustc,
                debug_assertions_requested,
                config.llvm_assertions,
            );

            debug = debug_toml;
            rustc_debug_assertions = rustc_debug_assertions_toml;
            std_debug_assertions = std_debug_assertions_toml;
            overflow_checks = overflow_checks_toml;
            overflow_checks_std = overflow_checks_std_toml;
            debug_logging = debug_logging_toml;
            debuginfo_level = debuginfo_level_toml;
            debuginfo_level_rustc = debuginfo_level_rustc_toml;
            debuginfo_level_std = debuginfo_level_std_toml;
            debuginfo_level_tools = debuginfo_level_tools_toml;
            debuginfo_level_tests = debuginfo_level_tests_toml;
            lld_enabled = lld_enabled_toml;
            std_features = std_features_toml;

            optimize = optimize_toml;
            config.rust_new_symbol_mangling = new_symbol_mangling;
            set(&mut config.rust_optimize_tests, optimize_tests);
            set(&mut config.codegen_tests, codegen_tests);
            set(&mut config.rust_rpath, rpath);
            set(&mut config.rust_strip, strip);
            set(&mut config.rust_frame_pointers, frame_pointers);
            config.rust_stack_protector = stack_protector;
            set(&mut config.jemalloc, jemalloc);
            set(&mut config.test_compare_mode, test_compare_mode);
            set(&mut config.backtrace, backtrace);
            if rust_description.is_some() {
                eprintln!(
                    "Warning: rust.description is deprecated. Use build.description instead."
                );
            }
            description = description.or(rust_description);
            set(&mut config.rust_dist_src, dist_src);
            set(&mut config.verbose_tests, verbose_tests);
            // in the case "false" is set explicitly, do not overwrite the command line args
            if let Some(true) = incremental {
                config.incremental = true;
            }
            set(&mut config.lld_mode, lld_mode);
            set(&mut config.llvm_bitcode_linker_enabled, llvm_bitcode_linker);

            config.rust_randomize_layout = randomize_layout.unwrap_or_default();
            config.llvm_tools_enabled = llvm_tools.unwrap_or(true);

            config.llvm_enzyme =
                llvm_enzyme.unwrap_or(config.channel == "dev" || config.channel == "nightly");
            config.rustc_default_linker = default_linker;
            config.musl_root = musl_root.map(PathBuf::from);
            config.save_toolstates = save_toolstates.map(PathBuf::from);
            set(
                &mut config.deny_warnings,
                match flags.warnings {
                    Warnings::Deny => Some(true),
                    Warnings::Warn => Some(false),
                    Warnings::Default => deny_warnings,
                },
            );
            set(&mut config.backtrace_on_ice, backtrace_on_ice);
            set(&mut config.rust_verify_llvm_ir, verify_llvm_ir);
            config.rust_thin_lto_import_instr_limit = thin_lto_import_instr_limit;
            set(&mut config.rust_remap_debuginfo, remap_debuginfo);
            set(&mut config.control_flow_guard, control_flow_guard);
            set(&mut config.ehcont_guard, ehcont_guard);
            config.llvm_libunwind_default =
                llvm_libunwind.map(|v| v.parse().expect("failed to parse rust.llvm-libunwind"));

            if let Some(ref backends) = codegen_backends {
                let available_backends = ["llvm", "cranelift", "gcc"];

                config.rust_codegen_backends = backends.iter().map(|s| {
                    if let Some(backend) = s.strip_prefix(CODEGEN_BACKEND_PREFIX) {
                        if available_backends.contains(&backend) {
                            panic!("Invalid value '{s}' for 'rust.codegen-backends'. Instead, please use '{backend}'.");
                        } else {
                            println!("HELP: '{s}' for 'rust.codegen-backends' might fail. \
                                Codegen backends are mostly defined without the '{CODEGEN_BACKEND_PREFIX}' prefix. \
                                In this case, it would be referred to as '{backend}'.");
                        }
                    }

                    s.clone()
                }).collect();
            }

            config.rust_codegen_units = codegen_units.map(threads_from_config);
            config.rust_codegen_units_std = codegen_units_std.map(threads_from_config);
            config.rust_profile_use = flags.rust_profile_use.or(profile_use);
            config.rust_profile_generate = flags.rust_profile_generate.or(profile_generate);
            config.rust_lto =
                lto.as_deref().map(|value| RustcLto::from_str(value).unwrap()).unwrap_or_default();
            config.rust_validate_mir_opts = validate_mir_opts;
        } else {
            config.rust_profile_use = flags.rust_profile_use;
            config.rust_profile_generate = flags.rust_profile_generate;
        }

        config.reproducible_artifacts = flags.reproducible_artifact;
        config.description = description;

        // We need to override `rust.channel` if it's manually specified when using the CI rustc.
        // This is because if the compiler uses a different channel than the one specified in bootstrap.toml,
        // tests may fail due to using a different channel than the one used by the compiler during tests.
        if let Some(commit) = &config.download_rustc_commit {
            if is_user_configured_rust_channel {
                println!(
                    "WARNING: `rust.download-rustc` is enabled. The `rust.channel` option will be overridden by the CI rustc's channel."
                );

                let channel = config
                    .read_file_by_commit(Path::new("src/ci/channel"), commit)
                    .trim()
                    .to_owned();

                config.channel = channel;
            }
        } else if config.rust_info.is_from_tarball() && !is_user_configured_rust_channel {
            ci_channel.clone_into(&mut config.channel);
        }

        if let Some(llvm) = toml.llvm {
            let Llvm {
                optimize: optimize_toml,
                thin_lto,
                release_debuginfo,
                assertions: _,
                tests,
                enzyme,
                plugins,
                ccache: llvm_ccache,
                static_libstdcpp,
                libzstd,
                ninja,
                targets,
                experimental_targets,
                link_jobs,
                link_shared,
                version_suffix,
                clang_cl,
                cflags,
                cxxflags,
                ldflags,
                use_libcxx,
                use_linker,
                allow_old_toolchain,
                offload,
                polly,
                clang,
                enable_warnings,
                download_ci_llvm,
                build_config,
            } = llvm;
            if llvm_ccache.is_some() {
                eprintln!("Warning: llvm.ccache is deprecated. Use build.ccache instead.");
            }

            ccache = ccache.or(llvm_ccache);
            set(&mut config.ninja_in_file, ninja);
            llvm_tests = tests;
            llvm_enzyme = enzyme;
            llvm_offload = offload;
            llvm_plugins = plugins;
            set(&mut config.llvm_optimize, optimize_toml);
            set(&mut config.llvm_thin_lto, thin_lto);
            set(&mut config.llvm_release_debuginfo, release_debuginfo);
            set(&mut config.llvm_static_stdcpp, static_libstdcpp);
            set(&mut config.llvm_libzstd, libzstd);
            if let Some(v) = link_shared {
                config.llvm_link_shared.set(Some(v));
            }
            config.llvm_targets.clone_from(&targets);
            config.llvm_experimental_targets.clone_from(&experimental_targets);
            config.llvm_link_jobs = link_jobs;
            config.llvm_version_suffix.clone_from(&version_suffix);
            config.llvm_clang_cl.clone_from(&clang_cl);

            config.llvm_cflags.clone_from(&cflags);
            config.llvm_cxxflags.clone_from(&cxxflags);
            config.llvm_ldflags.clone_from(&ldflags);
            set(&mut config.llvm_use_libcxx, use_libcxx);
            config.llvm_use_linker.clone_from(&use_linker);
            config.llvm_allow_old_toolchain = allow_old_toolchain.unwrap_or(false);
            config.llvm_offload = offload.unwrap_or(false);
            config.llvm_polly = polly.unwrap_or(false);
            config.llvm_clang = clang.unwrap_or(false);
            config.llvm_enable_warnings = enable_warnings.unwrap_or(false);
            config.llvm_build_config = build_config.clone().unwrap_or(Default::default());

            config.llvm_from_ci =
                config.parse_download_ci_llvm(download_ci_llvm, config.llvm_assertions);

            if config.llvm_from_ci {
                let warn = |option: &str| {
                    println!(
                        "WARNING: `{option}` will only be used on `compiler/rustc_llvm` build, not for the LLVM build."
                    );
                    println!(
                        "HELP: To use `{option}` for LLVM builds, set `download-ci-llvm` option to false."
                    );
                };

                if static_libstdcpp.is_some() {
                    warn("static-libstdcpp");
                }

                if link_shared.is_some() {
                    warn("link-shared");
                }

                // FIXME(#129153): instead of all the ad-hoc `download-ci-llvm` checks that follow,
                // use the `builder-config` present in tarballs since #128822 to compare the local
                // config to the ones used to build the LLVM artifacts on CI, and only notify users
                // if they've chosen a different value.

                if libzstd.is_some() {
                    println!(
                        "WARNING: when using `download-ci-llvm`, the local `llvm.libzstd` option, \
                        like almost all `llvm.*` options, will be ignored and set by the LLVM CI \
                        artifacts builder config."
                    );
                    println!(
                        "HELP: To use `llvm.libzstd` for LLVM/LLD builds, set `download-ci-llvm` option to false."
                    );
                }
            }

            if !config.llvm_from_ci && config.llvm_thin_lto && link_shared.is_none() {
                // If we're building with ThinLTO on, by default we want to link
                // to LLVM shared, to avoid re-doing ThinLTO (which happens in
                // the link step) with each stage.
                config.llvm_link_shared.set(Some(true));
            }
        } else {
            config.llvm_from_ci = config.parse_download_ci_llvm(None, false);
        }

        if let Some(gcc) = toml.gcc {
            config.gcc_ci_mode = match gcc.download_ci_gcc {
                Some(value) => match value {
                    true => GccCiMode::DownloadFromCi,
                    false => GccCiMode::BuildLocally,
                },
                None => GccCiMode::default(),
            };
        }

        if let Some(t) = toml.target {
            for (triple, cfg) in t {
                let mut target = Target::from_triple(&triple);

                if let Some(ref s) = cfg.llvm_config {
                    if config.download_rustc_commit.is_some() && triple == *config.build.triple {
                        panic!(
                            "setting llvm_config for the host is incompatible with download-rustc"
                        );
                    }
                    target.llvm_config = Some(config.src.join(s));
                }
                if let Some(patches) = cfg.llvm_has_rust_patches {
                    assert!(
                        config.submodules == Some(false) || cfg.llvm_config.is_some(),
                        "use of `llvm-has-rust-patches` is restricted to cases where either submodules are disabled or llvm-config been provided"
                    );
                    target.llvm_has_rust_patches = Some(patches);
                }
                if let Some(ref s) = cfg.llvm_filecheck {
                    target.llvm_filecheck = Some(config.src.join(s));
                }
                target.llvm_libunwind = cfg.llvm_libunwind.as_ref().map(|v| {
                    v.parse().unwrap_or_else(|_| {
                        panic!("failed to parse target.{triple}.llvm-libunwind")
                    })
                });
                if let Some(s) = cfg.no_std {
                    target.no_std = s;
                }
                target.cc = cfg.cc.map(PathBuf::from);
                target.cxx = cfg.cxx.map(PathBuf::from);
                target.ar = cfg.ar.map(PathBuf::from);
                target.ranlib = cfg.ranlib.map(PathBuf::from);
                target.linker = cfg.linker.map(PathBuf::from);
                target.crt_static = cfg.crt_static;
                target.musl_root = cfg.musl_root.map(PathBuf::from);
                target.musl_libdir = cfg.musl_libdir.map(PathBuf::from);
                target.wasi_root = cfg.wasi_root.map(PathBuf::from);
                target.qemu_rootfs = cfg.qemu_rootfs.map(PathBuf::from);
                target.runner = cfg.runner;
                target.sanitizers = cfg.sanitizers;
                target.profiler = cfg.profiler;
                target.rpath = cfg.rpath;
                target.optimized_compiler_builtins = cfg.optimized_compiler_builtins;
                target.jemalloc = cfg.jemalloc;

                if let Some(ref backends) = cfg.codegen_backends {
                    let available_backends = ["llvm", "cranelift", "gcc"];

                    target.codegen_backends = Some(backends.iter().map(|s| {
                        if let Some(backend) = s.strip_prefix(CODEGEN_BACKEND_PREFIX) {
                            if available_backends.contains(&backend) {
                                panic!("Invalid value '{s}' for 'target.{triple}.codegen-backends'. Instead, please use '{backend}'.");
                            } else {
                                println!("HELP: '{s}' for 'target.{triple}.codegen-backends' might fail. \
                                    Codegen backends are mostly defined without the '{CODEGEN_BACKEND_PREFIX}' prefix. \
                                    In this case, it would be referred to as '{backend}'.");
                            }
                        }

                        s.clone()
                    }).collect());
                }

                target.split_debuginfo = cfg.split_debuginfo.as_ref().map(|v| {
                    v.parse().unwrap_or_else(|_| {
                        panic!("invalid value for target.{triple}.split-debuginfo")
                    })
                });

                config.target_config.insert(TargetSelection::from_user(&triple), target);
            }
        }

        match ccache {
            Some(StringOrBool::String(ref s)) => config.ccache = Some(s.to_string()),
            Some(StringOrBool::Bool(true)) => {
                config.ccache = Some("ccache".to_string());
            }
            Some(StringOrBool::Bool(false)) | None => {}
        }

        if config.llvm_from_ci {
            let triple = &config.build.triple;
            let ci_llvm_bin = config.ci_llvm_root().join("bin");
            let build_target = config
                .target_config
                .entry(config.build)
                .or_insert_with(|| Target::from_triple(triple));

            check_ci_llvm!(build_target.llvm_config);
            check_ci_llvm!(build_target.llvm_filecheck);
            build_target.llvm_config = Some(ci_llvm_bin.join(exe("llvm-config", config.build)));
            build_target.llvm_filecheck = Some(ci_llvm_bin.join(exe("FileCheck", config.build)));
        }

        if let Some(dist) = toml.dist {
            let Dist {
                sign_folder,
                upload_addr,
                src_tarball,
                compression_formats,
                compression_profile,
                include_mingw_linker,
                vendor,
            } = dist;
            config.dist_sign_folder = sign_folder.map(PathBuf::from);
            config.dist_upload_addr = upload_addr;
            config.dist_compression_formats = compression_formats;
            set(&mut config.dist_compression_profile, compression_profile);
            set(&mut config.rust_dist_src, src_tarball);
            set(&mut config.dist_include_mingw_linker, include_mingw_linker);
            config.dist_vendor = vendor.unwrap_or_else(|| {
                // If we're building from git or tarball sources, enable it by default.
                config.rust_info.is_managed_git_subrepository()
                    || config.rust_info.is_from_tarball()
            });
        }

        if let Some(r) = rustfmt {
            *config.initial_rustfmt.borrow_mut() = if r.exists() {
                RustfmtState::SystemToolchain(r)
            } else {
                RustfmtState::Unavailable
            };
        }

        // Now that we've reached the end of our configuration, infer the
        // default values for all options that we haven't otherwise stored yet.

        config.llvm_tests = llvm_tests.unwrap_or(false);
        config.llvm_enzyme = llvm_enzyme.unwrap_or(false);
        config.llvm_offload = llvm_offload.unwrap_or(false);
        config.llvm_plugins = llvm_plugins.unwrap_or(false);
        config.rust_optimize = optimize.unwrap_or(RustOptimize::Bool(true));

        // We make `x86_64-unknown-linux-gnu` use the self-contained linker by default, so we will
        // build our internal lld and use it as the default linker, by setting the `rust.lld` config
        // to true by default:
        // - on the `x86_64-unknown-linux-gnu` target
        // - on the `dev` and `nightly` channels
        // - when building our in-tree llvm (i.e. the target has not set an `llvm-config`), so that
        //   we're also able to build the corresponding lld
        // - or when using an external llvm that's downloaded from CI, which also contains our prebuilt
        //   lld
        // - otherwise, we'd be using an external llvm, and lld would not necessarily available and
        //   thus, disabled
        // - similarly, lld will not be built nor used by default when explicitly asked not to, e.g.
        //   when the config sets `rust.lld = false`
        if config.build.triple == "x86_64-unknown-linux-gnu"
            && config.hosts == [config.build]
            && (config.channel == "dev" || config.channel == "nightly")
        {
            let no_llvm_config = config
                .target_config
                .get(&config.build)
                .is_some_and(|target_config| target_config.llvm_config.is_none());
            let enable_lld = config.llvm_from_ci || no_llvm_config;
            // Prefer the config setting in case an explicit opt-out is needed.
            config.lld_enabled = lld_enabled.unwrap_or(enable_lld);
        } else {
            set(&mut config.lld_enabled, lld_enabled);
        }

        if matches!(config.lld_mode, LldMode::SelfContained)
            && !config.lld_enabled
            && flags.stage.unwrap_or(0) > 0
        {
            panic!(
                "Trying to use self-contained lld as a linker, but LLD is not being added to the sysroot. Enable it with rust.lld = true."
            );
        }

        let default_std_features = BTreeSet::from([String::from("panic-unwind")]);
        config.rust_std_features = std_features.unwrap_or(default_std_features);

        let default = debug == Some(true);
        config.rustc_debug_assertions = rustc_debug_assertions.unwrap_or(default);
        config.std_debug_assertions = std_debug_assertions.unwrap_or(config.rustc_debug_assertions);
        config.rust_overflow_checks = overflow_checks.unwrap_or(default);
        config.rust_overflow_checks_std =
            overflow_checks_std.unwrap_or(config.rust_overflow_checks);

        config.rust_debug_logging = debug_logging.unwrap_or(config.rustc_debug_assertions);

        let with_defaults = |debuginfo_level_specific: Option<_>| {
            debuginfo_level_specific.or(debuginfo_level).unwrap_or(if debug == Some(true) {
                DebuginfoLevel::Limited
            } else {
                DebuginfoLevel::None
            })
        };
        config.rust_debuginfo_level_rustc = with_defaults(debuginfo_level_rustc);
        config.rust_debuginfo_level_std = with_defaults(debuginfo_level_std);
        config.rust_debuginfo_level_tools = with_defaults(debuginfo_level_tools);
        config.rust_debuginfo_level_tests = debuginfo_level_tests.unwrap_or(DebuginfoLevel::None);
        config.optimized_compiler_builtins =
            optimized_compiler_builtins.unwrap_or(config.channel != "dev");
        config.compiletest_diff_tool = compiletest_diff_tool;

        let download_rustc = config.download_rustc_commit.is_some();
        config.explicit_stage_from_cli = flags.stage.is_some();
        config.explicit_stage_from_config = test_stage.is_some()
            || build_stage.is_some()
            || doc_stage.is_some()
            || dist_stage.is_some()
            || install_stage.is_some()
            || check_stage.is_some()
            || bench_stage.is_some();
        // See https://github.com/rust-lang/compiler-team/issues/326
        config.stage = match config.cmd {
            Subcommand::Check { .. } => flags.stage.or(check_stage).unwrap_or(0),
            // `download-rustc` only has a speed-up for stage2 builds. Default to stage2 unless explicitly overridden.
            Subcommand::Doc { .. } => {
                flags.stage.or(doc_stage).unwrap_or(if download_rustc { 2 } else { 0 })
            }
            Subcommand::Build => {
                flags.stage.or(build_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Test { .. } | Subcommand::Miri { .. } => {
                flags.stage.or(test_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Bench { .. } => flags.stage.or(bench_stage).unwrap_or(2),
            Subcommand::Dist => flags.stage.or(dist_stage).unwrap_or(2),
            Subcommand::Install => flags.stage.or(install_stage).unwrap_or(2),
            Subcommand::Perf { .. } => flags.stage.unwrap_or(1),
            // These are all bootstrap tools, which don't depend on the compiler.
            // The stage we pass shouldn't matter, but use 0 just in case.
            Subcommand::Clean { .. }
            | Subcommand::Clippy { .. }
            | Subcommand::Fix
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. }
            | Subcommand::Suggest { .. }
            | Subcommand::Vendor { .. } => flags.stage.unwrap_or(0),
        };

        // CI should always run stage 2 builds, unless it specifically states otherwise
        #[cfg(not(test))]
        if flags.stage.is_none() && config.is_running_on_ci {
            match config.cmd {
                Subcommand::Test { .. }
                | Subcommand::Miri { .. }
                | Subcommand::Doc { .. }
                | Subcommand::Build
                | Subcommand::Bench { .. }
                | Subcommand::Dist
                | Subcommand::Install => {
                    assert_eq!(
                        config.stage, 2,
                        "x.py should be run with `--stage 2` on CI, but was run with `--stage {}`",
                        config.stage,
                    );
                }
                Subcommand::Clean { .. }
                | Subcommand::Check { .. }
                | Subcommand::Clippy { .. }
                | Subcommand::Fix
                | Subcommand::Run { .. }
                | Subcommand::Setup { .. }
                | Subcommand::Format { .. }
                | Subcommand::Suggest { .. }
                | Subcommand::Vendor { .. }
                | Subcommand::Perf { .. } => {}
            }
        }

        config
    }

    pub fn dry_run(&self) -> bool {
        match self.dry_run {
            DryRun::Disabled => false,
            DryRun::SelfCheck | DryRun::UserSelected => true,
        }
    }

    pub fn is_explicit_stage(&self) -> bool {
        self.explicit_stage_from_cli || self.explicit_stage_from_config
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    #[deprecated = "use `Builder::try_run` instead where possible"]
    pub(crate) fn try_run(&self, cmd: &mut Command) -> Result<(), ()> {
        if self.dry_run() {
            return Ok(());
        }
        self.verbose(|| println!("running: {cmd:?}"));
        build_helper::util::try_run(cmd, self.is_verbose())
    }

    pub(crate) fn test_args(&self) -> Vec<&str> {
        let mut test_args = match self.cmd {
            Subcommand::Test { ref test_args, .. }
            | Subcommand::Bench { ref test_args, .. }
            | Subcommand::Miri { ref test_args, .. } => {
                test_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        };
        test_args.extend(self.free_args.iter().map(|s| s.as_str()));
        test_args
    }

    pub(crate) fn args(&self) -> Vec<&str> {
        let mut args = match self.cmd {
            Subcommand::Run { ref args, .. } => {
                args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        };
        args.extend(self.free_args.iter().map(|s| s.as_str()));
        args
    }

    /// Returns the content of the given file at a specific commit.
    pub(crate) fn read_file_by_commit(&self, file: &Path, commit: &str) -> String {
        assert!(
            self.rust_info.is_managed_git_subrepository(),
            "`Config::read_file_by_commit` is not supported in non-git sources."
        );

        let mut git = helpers::git(Some(&self.src));
        git.arg("show").arg(format!("{commit}:{}", file.to_str().unwrap()));
        output(git.as_command_mut())
    }

    /// Bootstrap embeds a version number into the name of shared libraries it uploads in CI.
    /// Return the version it would have used for the given commit.
    pub(crate) fn artifact_version_part(&self, commit: &str) -> String {
        let (channel, version) = if self.rust_info.is_managed_git_subrepository() {
            let channel =
                self.read_file_by_commit(Path::new("src/ci/channel"), commit).trim().to_owned();
            let version =
                self.read_file_by_commit(Path::new("src/version"), commit).trim().to_owned();
            (channel, version)
        } else {
            let channel = fs::read_to_string(self.src.join("src/ci/channel"));
            let version = fs::read_to_string(self.src.join("src/version"));
            match (channel, version) {
                (Ok(channel), Ok(version)) => {
                    (channel.trim().to_owned(), version.trim().to_owned())
                }
                (channel, version) => {
                    let src = self.src.display();
                    eprintln!("ERROR: failed to determine artifact channel and/or version");
                    eprintln!(
                        "HELP: consider using a git checkout or ensure these files are readable"
                    );
                    if let Err(channel) = channel {
                        eprintln!("reading {src}/src/ci/channel failed: {channel:?}");
                    }
                    if let Err(version) = version {
                        eprintln!("reading {src}/src/version failed: {version:?}");
                    }
                    panic!();
                }
            }
        };

        match channel.as_str() {
            "stable" => version,
            "beta" => channel,
            "nightly" => channel,
            other => unreachable!("{:?} is not recognized as a valid channel", other),
        }
    }

    /// Try to find the relative path of `bindir`, otherwise return it in full.
    pub fn bindir_relative(&self) -> &Path {
        let bindir = &self.bindir;
        if bindir.is_absolute() {
            // Try to make it relative to the prefix.
            if let Some(prefix) = &self.prefix {
                if let Ok(stripped) = bindir.strip_prefix(prefix) {
                    return stripped;
                }
            }
        }
        bindir
    }

    /// Try to find the relative path of `libdir`.
    pub fn libdir_relative(&self) -> Option<&Path> {
        let libdir = self.libdir.as_ref()?;
        if libdir.is_relative() {
            Some(libdir)
        } else {
            // Try to make it relative to the prefix.
            libdir.strip_prefix(self.prefix.as_ref()?).ok()
        }
    }

    /// The absolute path to the downloaded LLVM artifacts.
    pub(crate) fn ci_llvm_root(&self) -> PathBuf {
        assert!(self.llvm_from_ci);
        self.out.join(self.build).join("ci-llvm")
    }

    /// Directory where the extracted `rustc-dev` component is stored.
    pub(crate) fn ci_rustc_dir(&self) -> PathBuf {
        assert!(self.download_rustc());
        self.out.join(self.build).join("ci-rustc")
    }

    /// Determine whether llvm should be linked dynamically.
    ///
    /// If `false`, llvm should be linked statically.
    /// This is computed on demand since LLVM might have to first be downloaded from CI.
    pub(crate) fn llvm_link_shared(&self) -> bool {
        let mut opt = self.llvm_link_shared.get();
        if opt.is_none() && self.dry_run() {
            // just assume static for now - dynamic linking isn't supported on all platforms
            return false;
        }

        let llvm_link_shared = *opt.get_or_insert_with(|| {
            if self.llvm_from_ci {
                self.maybe_download_ci_llvm();
                let ci_llvm = self.ci_llvm_root();
                let link_type = t!(
                    std::fs::read_to_string(ci_llvm.join("link-type.txt")),
                    format!("CI llvm missing: {}", ci_llvm.display())
                );
                link_type == "dynamic"
            } else {
                // unclear how thought-through this default is, but it maintains compatibility with
                // previous behavior
                false
            }
        });
        self.llvm_link_shared.set(opt);
        llvm_link_shared
    }

    /// Return whether we will use a downloaded, pre-compiled version of rustc, or just build from source.
    pub(crate) fn download_rustc(&self) -> bool {
        self.download_rustc_commit().is_some()
    }

    pub(crate) fn download_rustc_commit(&self) -> Option<&str> {
        static DOWNLOAD_RUSTC: OnceLock<Option<String>> = OnceLock::new();
        if self.dry_run() && DOWNLOAD_RUSTC.get().is_none() {
            // avoid trying to actually download the commit
            return self.download_rustc_commit.as_deref();
        }

        DOWNLOAD_RUSTC
            .get_or_init(|| match &self.download_rustc_commit {
                None => None,
                Some(commit) => {
                    self.download_ci_rustc(commit);

                    // CI-rustc can't be used without CI-LLVM. If `self.llvm_from_ci` is false, it means the "if-unchanged"
                    // logic has detected some changes in the LLVM submodule (download-ci-llvm=false can't happen here as
                    // we don't allow it while parsing the configuration).
                    if !self.llvm_from_ci {
                        // This happens when LLVM submodule is updated in CI, we should disable ci-rustc without an error
                        // to not break CI. For non-CI environments, we should return an error.
                        if self.is_running_on_ci {
                            println!("WARNING: LLVM submodule has changes, `download-rustc` will be disabled.");
                            return None;
                        } else {
                            panic!("ERROR: LLVM submodule has changes, `download-rustc` can't be used.");
                        }
                    }

                    if let Some(config_path) = &self.config {
                        let ci_config_toml = match self.get_builder_toml("ci-rustc") {
                            Ok(ci_config_toml) => ci_config_toml,
                            Err(e) if e.to_string().contains("unknown field") => {
                                println!("WARNING: CI rustc has some fields that are no longer supported in bootstrap; download-rustc will be disabled.");
                                println!("HELP: Consider rebasing to a newer commit if available.");
                                return None;
                            },
                            Err(e) => {
                                eprintln!("ERROR: Failed to parse CI rustc bootstrap.toml: {e}");
                                exit!(2);
                            },
                        };

                        let current_config_toml = Self::get_toml(config_path).unwrap();

                        // Check the config compatibility
                        // FIXME: this doesn't cover `--set` flags yet.
                        let res = check_incompatible_options_for_ci_rustc(
                            self.build,
                            current_config_toml,
                            ci_config_toml,
                        );

                        // Primarily used by CI runners to avoid handling download-rustc incompatible
                        // options one by one on shell scripts.
                        let disable_ci_rustc_if_incompatible = env::var_os("DISABLE_CI_RUSTC_IF_INCOMPATIBLE")
                            .is_some_and(|s| s == "1" || s == "true");

                        if disable_ci_rustc_if_incompatible && res.is_err() {
                            println!("WARNING: download-rustc is disabled with `DISABLE_CI_RUSTC_IF_INCOMPATIBLE` env.");
                            return None;
                        }

                        res.unwrap();
                    }

                    Some(commit.clone())
                }
            })
            .as_deref()
    }

    pub(crate) fn initial_rustfmt(&self) -> Option<PathBuf> {
        match &mut *self.initial_rustfmt.borrow_mut() {
            RustfmtState::SystemToolchain(p) | RustfmtState::Downloaded(p) => Some(p.clone()),
            RustfmtState::Unavailable => None,
            r @ RustfmtState::LazyEvaluated => {
                if self.dry_run() {
                    return Some(PathBuf::new());
                }
                let path = self.maybe_download_rustfmt();
                *r = if let Some(p) = &path {
                    RustfmtState::Downloaded(p.clone())
                } else {
                    RustfmtState::Unavailable
                };
                path
            }
        }
    }

    /// Runs a function if verbosity is greater than 0
    pub fn verbose(&self, f: impl Fn()) {
        if self.is_verbose() {
            f()
        }
    }

    pub fn sanitizers_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|t| t.sanitizers).unwrap_or(self.sanitizers)
    }

    pub fn needs_sanitizer_runtime_built(&self, target: TargetSelection) -> bool {
        // MSVC uses the Microsoft-provided sanitizer runtime, but all other runtimes we build.
        !target.is_msvc() && self.sanitizers_enabled(target)
    }

    pub fn any_sanitizers_to_build(&self) -> bool {
        self.target_config
            .iter()
            .any(|(ts, t)| !ts.is_msvc() && t.sanitizers.unwrap_or(self.sanitizers))
    }

    pub fn profiler_path(&self, target: TargetSelection) -> Option<&str> {
        match self.target_config.get(&target)?.profiler.as_ref()? {
            StringOrBool::String(s) => Some(s),
            StringOrBool::Bool(_) => None,
        }
    }

    pub fn profiler_enabled(&self, target: TargetSelection) -> bool {
        self.target_config
            .get(&target)
            .and_then(|t| t.profiler.as_ref())
            .map(StringOrBool::is_string_or_true)
            .unwrap_or(self.profiler)
    }

    pub fn any_profiler_enabled(&self) -> bool {
        self.target_config.values().any(|t| matches!(&t.profiler, Some(p) if p.is_string_or_true()))
            || self.profiler
    }

    pub fn rpath_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|t| t.rpath).unwrap_or(self.rust_rpath)
    }

    pub fn optimized_compiler_builtins(&self, target: TargetSelection) -> bool {
        self.target_config
            .get(&target)
            .and_then(|t| t.optimized_compiler_builtins)
            .unwrap_or(self.optimized_compiler_builtins)
    }

    pub fn llvm_enabled(&self, target: TargetSelection) -> bool {
        self.codegen_backends(target).contains(&"llvm".to_owned())
    }

    pub fn llvm_libunwind(&self, target: TargetSelection) -> LlvmLibunwind {
        self.target_config
            .get(&target)
            .and_then(|t| t.llvm_libunwind)
            .or(self.llvm_libunwind_default)
            .unwrap_or(if target.contains("fuchsia") {
                LlvmLibunwind::InTree
            } else {
                LlvmLibunwind::No
            })
    }

    pub fn split_debuginfo(&self, target: TargetSelection) -> SplitDebuginfo {
        self.target_config
            .get(&target)
            .and_then(|t| t.split_debuginfo)
            .unwrap_or_else(|| SplitDebuginfo::default_for_platform(target))
    }

    /// Returns whether or not submodules should be managed by bootstrap.
    pub fn submodules(&self) -> bool {
        // If not specified in config, the default is to only manage
        // submodules if we're currently inside a git repository.
        self.submodules.unwrap_or(self.rust_info.is_managed_git_subrepository())
    }

    pub fn codegen_backends(&self, target: TargetSelection) -> &[String] {
        self.target_config
            .get(&target)
            .and_then(|cfg| cfg.codegen_backends.as_deref())
            .unwrap_or(&self.rust_codegen_backends)
    }

    pub fn jemalloc(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|cfg| cfg.jemalloc).unwrap_or(self.jemalloc)
    }

    pub fn default_codegen_backend(&self, target: TargetSelection) -> Option<String> {
        self.codegen_backends(target).first().cloned()
    }

    pub fn git_config(&self) -> GitConfig<'_> {
        GitConfig {
            git_repository: &self.stage0_metadata.config.git_repository,
            nightly_branch: &self.stage0_metadata.config.nightly_branch,
            git_merge_commit_email: &self.stage0_metadata.config.git_merge_commit_email,
        }
    }

    /// Given a path to the directory of a submodule, update it.
    ///
    /// `relative_path` should be relative to the root of the git repository, not an absolute path.
    ///
    /// This *does not* update the submodule if `bootstrap.toml` explicitly says
    /// not to, or if we're not in a git repository (like a plain source
    /// tarball). Typically [`crate::Build::require_submodule`] should be
    /// used instead to provide a nice error to the user if the submodule is
    /// missing.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "Config::update_submodule",
            skip_all,
            fields(relative_path = ?relative_path),
        ),
    )]
    pub(crate) fn update_submodule(&self, relative_path: &str) {
        if self.rust_info.is_from_tarball() || !self.submodules() {
            return;
        }

        let absolute_path = self.src.join(relative_path);

        // NOTE: The check for the empty directory is here because when running x.py the first time,
        // the submodule won't be checked out. Check it out now so we can build it.
        if !GitInfo::new(false, &absolute_path).is_managed_git_subrepository()
            && !helpers::dir_is_empty(&absolute_path)
        {
            return;
        }

        // Submodule updating actually happens during in the dry run mode. We need to make sure that
        // all the git commands below are actually executed, because some follow-up code
        // in bootstrap might depend on the submodules being checked out. Furthermore, not all
        // the command executions below work with an empty output (produced during dry run).
        // Therefore, all commands below are marked with `run_always()`, so that they also run in
        // dry run mode.
        let submodule_git = || {
            let mut cmd = helpers::git(Some(&absolute_path));
            cmd.run_always();
            cmd
        };

        // Determine commit checked out in submodule.
        let checked_out_hash = output(submodule_git().args(["rev-parse", "HEAD"]).as_command_mut());
        let checked_out_hash = checked_out_hash.trim_end();
        // Determine commit that the submodule *should* have.
        let recorded = output(
            helpers::git(Some(&self.src))
                .run_always()
                .args(["ls-tree", "HEAD"])
                .arg(relative_path)
                .as_command_mut(),
        );

        let actual_hash = recorded
            .split_whitespace()
            .nth(2)
            .unwrap_or_else(|| panic!("unexpected output `{}`", recorded));

        if actual_hash == checked_out_hash {
            // already checked out
            return;
        }

        println!("Updating submodule {relative_path}");
        self.check_run(
            helpers::git(Some(&self.src))
                .run_always()
                .args(["submodule", "-q", "sync"])
                .arg(relative_path),
        );

        // Try passing `--progress` to start, then run git again without if that fails.
        let update = |progress: bool| {
            // Git is buggy and will try to fetch submodules from the tracking branch for *this* repository,
            // even though that has no relation to the upstream for the submodule.
            let current_branch = output_result(
                helpers::git(Some(&self.src))
                    .allow_failure()
                    .run_always()
                    .args(["symbolic-ref", "--short", "HEAD"])
                    .as_command_mut(),
            )
            .map(|b| b.trim().to_owned());

            let mut git = helpers::git(Some(&self.src)).allow_failure();
            git.run_always();
            if let Ok(branch) = current_branch {
                // If there is a tag named after the current branch, git will try to disambiguate by prepending `heads/` to the branch name.
                // This syntax isn't accepted by `branch.{branch}`. Strip it.
                let branch = branch.strip_prefix("heads/").unwrap_or(&branch);
                git.arg("-c").arg(format!("branch.{branch}.remote=origin"));
            }
            git.args(["submodule", "update", "--init", "--recursive", "--depth=1"]);
            if progress {
                git.arg("--progress");
            }
            git.arg(relative_path);
            git
        };
        if !self.check_run(&mut update(true)) {
            self.check_run(&mut update(false));
        }

        // Save any local changes, but avoid running `git stash pop` if there are none (since it will exit with an error).
        // diff-index reports the modifications through the exit status
        let has_local_modifications = !self.check_run(submodule_git().allow_failure().args([
            "diff-index",
            "--quiet",
            "HEAD",
        ]));
        if has_local_modifications {
            self.check_run(submodule_git().args(["stash", "push"]));
        }

        self.check_run(submodule_git().args(["reset", "-q", "--hard"]));
        self.check_run(submodule_git().args(["clean", "-qdfx"]));

        if has_local_modifications {
            self.check_run(submodule_git().args(["stash", "pop"]));
        }
    }

    #[cfg(test)]
    pub fn check_stage0_version(&self, _program_path: &Path, _component_name: &'static str) {}

    /// check rustc/cargo version is same or lower with 1 apart from the building one
    #[cfg(not(test))]
    pub fn check_stage0_version(&self, program_path: &Path, component_name: &'static str) {
        use build_helper::util::fail;

        if self.dry_run() {
            return;
        }

        let stage0_output = output(Command::new(program_path).arg("--version"));
        let mut stage0_output = stage0_output.lines().next().unwrap().split(' ');

        let stage0_name = stage0_output.next().unwrap();
        if stage0_name != component_name {
            fail(&format!(
                "Expected to find {component_name} at {} but it claims to be {stage0_name}",
                program_path.display()
            ));
        }

        let stage0_version =
            semver::Version::parse(stage0_output.next().unwrap().split('-').next().unwrap().trim())
                .unwrap();
        let source_version = semver::Version::parse(
            fs::read_to_string(self.src.join("src/version")).unwrap().trim(),
        )
        .unwrap();
        if !(source_version == stage0_version
            || (source_version.major == stage0_version.major
                && (source_version.minor == stage0_version.minor
                    || source_version.minor == stage0_version.minor + 1)))
        {
            let prev_version = format!("{}.{}.x", source_version.major, source_version.minor - 1);
            fail(&format!(
                "Unexpected {component_name} version: {stage0_version}, we should use {prev_version}/{source_version} to build source with {source_version}"
            ));
        }
    }

    /// Returns the commit to download, or `None` if we shouldn't download CI artifacts.
    fn download_ci_rustc_commit(
        &self,
        download_rustc: Option<StringOrBool>,
        debug_assertions_requested: bool,
        llvm_assertions: bool,
    ) -> Option<String> {
        if !is_download_ci_available(&self.build.triple, llvm_assertions) {
            return None;
        }

        // If `download-rustc` is not set, default to rebuilding.
        let if_unchanged = match download_rustc {
            // Globally default `download-rustc` to `false`, because some contributors don't use
            // profiles for reasons such as:
            // - They need to seamlessly switch between compiler/library work.
            // - They don't want to use compiler profile because they need to override too many
            //   things and it's easier to not use a profile.
            None | Some(StringOrBool::Bool(false)) => return None,
            Some(StringOrBool::Bool(true)) => false,
            Some(StringOrBool::String(s)) if s == "if-unchanged" => {
                if !self.rust_info.is_managed_git_subrepository() {
                    println!(
                        "ERROR: `download-rustc=if-unchanged` is only compatible with Git managed sources."
                    );
                    crate::exit!(1);
                }

                true
            }
            Some(StringOrBool::String(other)) => {
                panic!("unrecognized option for download-rustc: {other}")
            }
        };

        // RUSTC_IF_UNCHANGED_ALLOWED_PATHS
        let mut allowed_paths = RUSTC_IF_UNCHANGED_ALLOWED_PATHS.to_vec();

        // In CI, disable ci-rustc if there are changes in the library tree. But for non-CI, allow
        // these changes to speed up the build process for library developers. This provides consistent
        // functionality for library developers between `download-rustc=true` and `download-rustc="if-unchanged"`
        // options.
        //
        // If you update "library" logic here, update `builder::tests::ci_rustc_if_unchanged_logic` test
        // logic accordingly.
        if !self.is_running_on_ci {
            allowed_paths.push(":!library");
        }

        let commit = if self.rust_info.is_managed_git_subrepository() {
            // Look for a version to compare to based on the current commit.
            // Only commits merged by bors will have CI artifacts.
            match self.last_modified_commit(&allowed_paths, "download-rustc", if_unchanged) {
                Some(commit) => commit,
                None => {
                    if if_unchanged {
                        return None;
                    }
                    println!("ERROR: could not find commit hash for downloading rustc");
                    println!("HELP: maybe your repository history is too shallow?");
                    println!(
                        "HELP: consider setting `rust.download-rustc=false` in bootstrap.toml"
                    );
                    println!("HELP: or fetch enough history to include one upstream commit");
                    crate::exit!(1);
                }
            }
        } else {
            channel::read_commit_info_file(&self.src)
                .map(|info| info.sha.trim().to_owned())
                .expect("git-commit-info is missing in the project root")
        };

        if self.is_running_on_ci && {
            let head_sha =
                output(helpers::git(Some(&self.src)).arg("rev-parse").arg("HEAD").as_command_mut());
            let head_sha = head_sha.trim();
            commit == head_sha
        } {
            eprintln!("CI rustc commit matches with HEAD and we are in CI.");
            eprintln!(
                "`rustc.download-ci` functionality will be skipped as artifacts are not available."
            );
            return None;
        }

        if debug_assertions_requested {
            eprintln!(
                "WARN: `rust.debug-assertions = true` will prevent downloading CI rustc as alt CI \
                rustc is not currently built with debug assertions."
            );
            return None;
        }

        Some(commit)
    }

    fn parse_download_ci_llvm(
        &self,
        download_ci_llvm: Option<StringOrBool>,
        asserts: bool,
    ) -> bool {
        // We don't ever want to use `true` on CI, as we should not
        // download upstream artifacts if there are any local modifications.
        let default = if self.is_running_on_ci {
            StringOrBool::String("if-unchanged".to_string())
        } else {
            StringOrBool::Bool(true)
        };
        let download_ci_llvm = download_ci_llvm.unwrap_or(default);

        let if_unchanged = || {
            if self.rust_info.is_from_tarball() {
                // Git is needed for running "if-unchanged" logic.
                println!("ERROR: 'if-unchanged' is only compatible with Git managed sources.");
                crate::exit!(1);
            }

            // Fetching the LLVM submodule is unnecessary for self-tests.
            #[cfg(not(test))]
            self.update_submodule("src/llvm-project");

            // Check for untracked changes in `src/llvm-project` and other important places.
            let has_changes = self
                .last_modified_commit(LLVM_INVALIDATION_PATHS, "download-ci-llvm", true)
                .is_none();

            // Return false if there are untracked changes, otherwise check if CI LLVM is available.
            if has_changes { false } else { llvm::is_ci_llvm_available_for_target(self, asserts) }
        };

        match download_ci_llvm {
            StringOrBool::Bool(b) => {
                if !b && self.download_rustc_commit.is_some() {
                    panic!(
                        "`llvm.download-ci-llvm` cannot be set to `false` if `rust.download-rustc` is set to `true` or `if-unchanged`."
                    );
                }

                if b && self.is_running_on_ci {
                    // On CI, we must always rebuild LLVM if there were any modifications to it
                    panic!(
                        "`llvm.download-ci-llvm` cannot be set to `true` on CI. Use `if-unchanged` instead."
                    );
                }

                // If download-ci-llvm=true we also want to check that CI llvm is available
                b && llvm::is_ci_llvm_available_for_target(self, asserts)
            }
            StringOrBool::String(s) if s == "if-unchanged" => if_unchanged(),
            StringOrBool::String(other) => {
                panic!("unrecognized option for download-ci-llvm: {:?}", other)
            }
        }
    }

    /// Returns the last commit in which any of `modified_paths` were changed,
    /// or `None` if there are untracked changes in the working directory and `if_unchanged` is true.
    pub fn last_modified_commit(
        &self,
        modified_paths: &[&str],
        option_name: &str,
        if_unchanged: bool,
    ) -> Option<String> {
        assert!(
            self.rust_info.is_managed_git_subrepository(),
            "Can't run `Config::last_modified_commit` on a non-git source."
        );

        // Look for a version to compare to based on the current commit.
        // Only commits merged by bors will have CI artifacts.
        let commit = get_closest_merge_commit(Some(&self.src), &self.git_config(), &[]).unwrap();
        if commit.is_empty() {
            println!("error: could not find commit hash for downloading components from CI");
            println!("help: maybe your repository history is too shallow?");
            println!("help: consider disabling `{option_name}`");
            println!("help: or fetch enough history to include one upstream commit");
            crate::exit!(1);
        }

        // Warn if there were changes to the compiler or standard library since the ancestor commit.
        let mut git = helpers::git(Some(&self.src));
        git.args(["diff-index", "--quiet", &commit, "--"]).args(modified_paths);

        let has_changes = !t!(git.as_command_mut().status()).success();
        if has_changes {
            if if_unchanged {
                if self.is_verbose() {
                    println!(
                        "warning: saw changes to one of {modified_paths:?} since {commit}; \
                            ignoring `{option_name}`"
                    );
                }
                return None;
            }
            println!(
                "warning: `{option_name}` is enabled, but there are changes to one of {modified_paths:?}"
            );
        }

        Some(commit.to_string())
    }
}

/// Compares the current `Llvm` options against those in the CI LLVM builder and detects any incompatible options.
/// It does this by destructuring the `Llvm` instance to make sure every `Llvm` field is covered and not missing.
#[cfg(not(test))]
pub(crate) fn check_incompatible_options_for_ci_llvm(
    current_config_toml: TomlConfig,
    ci_config_toml: TomlConfig,
) -> Result<(), String> {
    macro_rules! err {
        ($current:expr, $expected:expr) => {
            if let Some(current) = &$current {
                if Some(current) != $expected.as_ref() {
                    return Err(format!(
                        "ERROR: Setting `llvm.{}` is incompatible with `llvm.download-ci-llvm`. \
                        Current value: {:?}, Expected value(s): {}{:?}",
                        stringify!($expected).replace("_", "-"),
                        $current,
                        if $expected.is_some() { "None/" } else { "" },
                        $expected,
                    ));
                };
            };
        };
    }

    macro_rules! warn {
        ($current:expr, $expected:expr) => {
            if let Some(current) = &$current {
                if Some(current) != $expected.as_ref() {
                    println!(
                        "WARNING: `llvm.{}` has no effect with `llvm.download-ci-llvm`. \
                        Current value: {:?}, Expected value(s): {}{:?}",
                        stringify!($expected).replace("_", "-"),
                        $current,
                        if $expected.is_some() { "None/" } else { "" },
                        $expected,
                    );
                };
            };
        };
    }

    let (Some(current_llvm_config), Some(ci_llvm_config)) =
        (current_config_toml.llvm, ci_config_toml.llvm)
    else {
        return Ok(());
    };

    let Llvm {
        optimize,
        thin_lto,
        release_debuginfo,
        assertions: _,
        tests: _,
        plugins,
        ccache: _,
        static_libstdcpp: _,
        libzstd,
        ninja: _,
        targets,
        experimental_targets,
        link_jobs: _,
        link_shared: _,
        version_suffix,
        clang_cl,
        cflags,
        cxxflags,
        ldflags,
        use_libcxx,
        use_linker,
        allow_old_toolchain,
        offload,
        polly,
        clang,
        enable_warnings,
        download_ci_llvm: _,
        build_config,
        enzyme,
    } = ci_llvm_config;

    err!(current_llvm_config.optimize, optimize);
    err!(current_llvm_config.thin_lto, thin_lto);
    err!(current_llvm_config.release_debuginfo, release_debuginfo);
    err!(current_llvm_config.libzstd, libzstd);
    err!(current_llvm_config.targets, targets);
    err!(current_llvm_config.experimental_targets, experimental_targets);
    err!(current_llvm_config.clang_cl, clang_cl);
    err!(current_llvm_config.version_suffix, version_suffix);
    err!(current_llvm_config.cflags, cflags);
    err!(current_llvm_config.cxxflags, cxxflags);
    err!(current_llvm_config.ldflags, ldflags);
    err!(current_llvm_config.use_libcxx, use_libcxx);
    err!(current_llvm_config.use_linker, use_linker);
    err!(current_llvm_config.allow_old_toolchain, allow_old_toolchain);
    err!(current_llvm_config.offload, offload);
    err!(current_llvm_config.polly, polly);
    err!(current_llvm_config.clang, clang);
    err!(current_llvm_config.build_config, build_config);
    err!(current_llvm_config.plugins, plugins);
    err!(current_llvm_config.enzyme, enzyme);

    warn!(current_llvm_config.enable_warnings, enable_warnings);

    Ok(())
}

/// Compares the current Rust options against those in the CI rustc builder and detects any incompatible options.
/// It does this by destructuring the `Rust` instance to make sure every `Rust` field is covered and not missing.
fn check_incompatible_options_for_ci_rustc(
    host: TargetSelection,
    current_config_toml: TomlConfig,
    ci_config_toml: TomlConfig,
) -> Result<(), String> {
    macro_rules! err {
        ($current:expr, $expected:expr, $config_section:expr) => {
            if let Some(current) = &$current {
                if Some(current) != $expected.as_ref() {
                    return Err(format!(
                        "ERROR: Setting `{}` is incompatible with `rust.download-rustc`. \
                        Current value: {:?}, Expected value(s): {}{:?}",
                        format!("{}.{}", $config_section, stringify!($expected).replace("_", "-")),
                        $current,
                        if $expected.is_some() { "None/" } else { "" },
                        $expected,
                    ));
                };
            };
        };
    }

    macro_rules! warn {
        ($current:expr, $expected:expr, $config_section:expr) => {
            if let Some(current) = &$current {
                if Some(current) != $expected.as_ref() {
                    println!(
                        "WARNING: `{}` has no effect with `rust.download-rustc`. \
                        Current value: {:?}, Expected value(s): {}{:?}",
                        format!("{}.{}", $config_section, stringify!($expected).replace("_", "-")),
                        $current,
                        if $expected.is_some() { "None/" } else { "" },
                        $expected,
                    );
                };
            };
        };
    }

    let current_profiler = current_config_toml.build.as_ref().and_then(|b| b.profiler);
    let profiler = ci_config_toml.build.as_ref().and_then(|b| b.profiler);
    err!(current_profiler, profiler, "build");

    let current_optimized_compiler_builtins =
        current_config_toml.build.as_ref().and_then(|b| b.optimized_compiler_builtins);
    let optimized_compiler_builtins =
        ci_config_toml.build.as_ref().and_then(|b| b.optimized_compiler_builtins);
    err!(current_optimized_compiler_builtins, optimized_compiler_builtins, "build");

    // We always build the in-tree compiler on cross targets, so we only care
    // about the host target here.
    let host_str = host.to_string();
    if let Some(current_cfg) = current_config_toml.target.as_ref().and_then(|c| c.get(&host_str)) {
        if current_cfg.profiler.is_some() {
            let ci_target_toml = ci_config_toml.target.as_ref().and_then(|c| c.get(&host_str));
            let ci_cfg = ci_target_toml.ok_or(format!(
                "Target specific config for '{host_str}' is not present for CI-rustc"
            ))?;

            let profiler = &ci_cfg.profiler;
            err!(current_cfg.profiler, profiler, "build");

            let optimized_compiler_builtins = &ci_cfg.optimized_compiler_builtins;
            err!(current_cfg.optimized_compiler_builtins, optimized_compiler_builtins, "build");
        }
    }

    let (Some(current_rust_config), Some(ci_rust_config)) =
        (current_config_toml.rust, ci_config_toml.rust)
    else {
        return Ok(());
    };

    let Rust {
        // Following options are the CI rustc incompatible ones.
        optimize,
        randomize_layout,
        debug_logging,
        debuginfo_level_rustc,
        llvm_tools,
        llvm_bitcode_linker,
        lto,
        stack_protector,
        strip,
        lld_mode,
        jemalloc,
        rpath,
        channel,
        description,
        incremental,
        default_linker,
        std_features,

        // Rest of the options can simply be ignored.
        debug: _,
        codegen_units: _,
        codegen_units_std: _,
        rustc_debug_assertions: _,
        std_debug_assertions: _,
        overflow_checks: _,
        overflow_checks_std: _,
        debuginfo_level: _,
        debuginfo_level_std: _,
        debuginfo_level_tools: _,
        debuginfo_level_tests: _,
        backtrace: _,
        musl_root: _,
        verbose_tests: _,
        optimize_tests: _,
        codegen_tests: _,
        omit_git_hash: _,
        dist_src: _,
        save_toolstates: _,
        codegen_backends: _,
        lld: _,
        deny_warnings: _,
        backtrace_on_ice: _,
        verify_llvm_ir: _,
        thin_lto_import_instr_limit: _,
        remap_debuginfo: _,
        test_compare_mode: _,
        llvm_libunwind: _,
        control_flow_guard: _,
        ehcont_guard: _,
        new_symbol_mangling: _,
        profile_generate: _,
        profile_use: _,
        download_rustc: _,
        validate_mir_opts: _,
        frame_pointers: _,
    } = ci_rust_config;

    // There are two kinds of checks for CI rustc incompatible options:
    //    1. Checking an option that may change the compiler behaviour/output.
    //    2. Checking an option that have no effect on the compiler behaviour/output.
    //
    // If the option belongs to the first category, we call `err` macro for a hard error;
    // otherwise, we just print a warning with `warn` macro.

    err!(current_rust_config.optimize, optimize, "rust");
    err!(current_rust_config.randomize_layout, randomize_layout, "rust");
    err!(current_rust_config.debug_logging, debug_logging, "rust");
    err!(current_rust_config.debuginfo_level_rustc, debuginfo_level_rustc, "rust");
    err!(current_rust_config.rpath, rpath, "rust");
    err!(current_rust_config.strip, strip, "rust");
    err!(current_rust_config.lld_mode, lld_mode, "rust");
    err!(current_rust_config.llvm_tools, llvm_tools, "rust");
    err!(current_rust_config.llvm_bitcode_linker, llvm_bitcode_linker, "rust");
    err!(current_rust_config.jemalloc, jemalloc, "rust");
    err!(current_rust_config.default_linker, default_linker, "rust");
    err!(current_rust_config.stack_protector, stack_protector, "rust");
    err!(current_rust_config.lto, lto, "rust");
    err!(current_rust_config.std_features, std_features, "rust");

    warn!(current_rust_config.channel, channel, "rust");
    warn!(current_rust_config.description, description, "rust");
    warn!(current_rust_config.incremental, incremental, "rust");

    Ok(())
}

fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}

fn threads_from_config(v: u32) -> u32 {
    match v {
        0 => std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as u32,
        n => n,
    }
}
