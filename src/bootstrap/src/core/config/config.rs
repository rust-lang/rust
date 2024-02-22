//! Serialized configuration of a build.
//!
//! This module implements parsing `config.toml` configuration files to tweak
//! how the build runs.

use std::cell::{Cell, RefCell};
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt::{self, Display};
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::OnceLock;

use crate::core::build_steps::compile::CODEGEN_BACKEND_PREFIX;
use crate::core::build_steps::llvm;
use crate::core::config::flags::{Color, Flags, Warnings};
use crate::utils::cache::{Interned, INTERNER};
use crate::utils::channel::{self, GitInfo};
use crate::utils::helpers::{exe, output, t};
use build_helper::exit;
use semver::Version;
use serde::{Deserialize, Deserializer};
use serde_derive::Deserialize;

pub use crate::core::config::flags::Subcommand;
use build_helper::git::GitConfig;

macro_rules! check_ci_llvm {
    ($name:expr) => {
        assert!(
            $name.is_none(),
            "setting {} is incompatible with download-ci-llvm.",
            stringify!($name)
        );
    };
}

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

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub enum DebuginfoLevel {
    #[default]
    None,
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
            StringOrInt::String("none") | StringOrInt::Int(0) => DebuginfoLevel::None,
            StringOrInt::String("line-tables-only") => DebuginfoLevel::LineTablesOnly,
            StringOrInt::String("limited") | StringOrInt::Int(1) => DebuginfoLevel::Limited,
            StringOrInt::String("full") | StringOrInt::Int(2) => DebuginfoLevel::Full,
            StringOrInt::Int(n) => {
                let other = serde::de::Unexpected::Signed(n);
                return Err(D::Error::invalid_value(other, &"expected 0, 1, or 2"));
            }
            StringOrInt::String(s) => {
                let other = serde::de::Unexpected::Str(s);
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
#[derive(Default, Copy, Clone)]
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

/// Global configuration for the entire build and/or bootstrap.
///
/// This structure is parsed from `config.toml`, and some of the fields are inferred from `git` or build-time parameters.
///
/// Note that this structure is not decoded directly into, but rather it is
/// filled out from the decoded forms of the structs below. For documentation
/// each field, see the corresponding fields in
/// `config.example.toml`.
#[derive(Default, Clone)]
pub struct Config {
    pub changelog_seen: Option<usize>, // FIXME: Deprecated field. Remove it at 2024.
    pub change_id: Option<usize>,
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
    pub stage0_metadata: Stage0Metadata,
    pub android_ndk: Option<PathBuf>,
    /// Whether to use the `c` feature of the `compiler_builtins` crate.
    pub optimized_compiler_builtins: bool,

    pub stdout_is_tty: bool,
    pub stderr_is_tty: bool,

    pub on_fail: Option<String>,
    pub stage: u32,
    pub keep_stage: Vec<u32>,
    pub keep_stage_std: Vec<u32>,
    pub src: PathBuf,
    /// defaults to `config.toml`
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
    pub llvm_plugins: bool,
    pub llvm_optimize: bool,
    pub llvm_thin_lto: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_static_stdcpp: bool,
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

    pub llvm_cflags: Option<String>,
    pub llvm_cxxflags: Option<String>,
    pub llvm_ldflags: Option<String>,
    pub llvm_use_libcxx: bool,

    // rust codegen options
    pub rust_optimize: RustOptimize,
    pub rust_codegen_units: Option<u32>,
    pub rust_codegen_units_std: Option<u32>,
    pub rust_debug_assertions: bool,
    pub rust_debug_assertions_std: bool,
    pub rust_overflow_checks: bool,
    pub rust_overflow_checks_std: bool,
    pub rust_debug_logging: bool,
    pub rust_debuginfo_level_rustc: DebuginfoLevel,
    pub rust_debuginfo_level_std: DebuginfoLevel,
    pub rust_debuginfo_level_tools: DebuginfoLevel,
    pub rust_debuginfo_level_tests: DebuginfoLevel,
    pub rust_split_debuginfo: SplitDebuginfo,
    pub rust_rpath: bool,
    pub rust_strip: bool,
    pub rust_frame_pointers: bool,
    pub rust_stack_protector: Option<String>,
    pub rustc_parallel: bool,
    pub rustc_default_linker: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_dist_src: bool,
    pub rust_codegen_backends: Vec<Interned<String>>,
    pub rust_verify_llvm_ir: bool,
    pub rust_thin_lto_import_instr_limit: Option<u32>,
    pub rust_remap_debuginfo: bool,
    pub rust_new_symbol_mangling: Option<bool>,
    pub rust_profile_use: Option<String>,
    pub rust_profile_generate: Option<String>,
    pub rust_lto: RustcLto,
    pub rust_validate_mir_opts: Option<u32>,
    pub llvm_profile_use: Option<String>,
    pub llvm_profile_generate: bool,
    pub llvm_libunwind_default: Option<LlvmLibunwind>,
    pub enable_bolt_settings: bool,

    pub reproducible_artifacts: Vec<String>,

    pub build: TargetSelection,
    pub hosts: Vec<TargetSelection>,
    pub targets: Vec<TargetSelection>,
    pub local_rebuild: bool,
    pub jemalloc: bool,
    pub control_flow_guard: bool,
    pub ehcont_guard: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_compression_formats: Option<Vec<String>>,
    pub dist_compression_profile: String,
    pub dist_include_mingw_linker: bool,

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
    pub missing_tools: bool, // FIXME: Deprecated field. Remove it at 2024.

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
    pub python: Option<PathBuf>,
    pub reuse: Option<PathBuf>,
    pub cargo_native_static: bool,
    pub configure_args: Vec<String>,
    pub out: PathBuf,
    pub rust_info: channel::GitInfo,

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,

    #[cfg(not(test))]
    initial_rustfmt: RefCell<RustfmtState>,
    #[cfg(test)]
    pub initial_rustfmt: RefCell<RustfmtState>,

    pub paths: Vec<PathBuf>,
}

#[derive(Default, Deserialize, Clone)]
pub struct Stage0Metadata {
    pub compiler: CompilerMetadata,
    pub config: Stage0Config,
    pub checksums_sha256: HashMap<String, String>,
    pub rustfmt: Option<RustfmtMetadata>,
}
#[derive(Default, Deserialize, Clone)]
pub struct CompilerMetadata {
    pub date: String,
    pub version: String,
}

#[derive(Default, Deserialize, Clone)]
pub struct Stage0Config {
    pub dist_server: String,
    pub artifacts_server: String,
    pub artifacts_with_llvm_assertions_server: String,
    pub git_merge_commit_email: String,
    pub git_repository: String,
    pub nightly_branch: String,
}
#[derive(Default, Deserialize, Clone)]
pub struct RustfmtMetadata {
    pub date: String,
    pub version: String,
}

#[derive(Clone, Debug, Default)]
pub enum RustfmtState {
    SystemToolchain(PathBuf),
    Downloaded(PathBuf),
    Unavailable,
    #[default]
    LazyEvaluated,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
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

#[derive(Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// `rust.split-debuginfo` in `config.example.toml`.
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

/// Per-target configuration stored in the global configuration structure.
#[derive(Default, Clone)]
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
    pub sanitizers: Option<bool>,
    pub profiler: Option<StringOrBool>,
    pub rpath: Option<bool>,
    pub crt_static: Option<bool>,
    pub musl_root: Option<PathBuf>,
    pub musl_libdir: Option<PathBuf>,
    pub wasi_root: Option<PathBuf>,
    pub qemu_rootfs: Option<PathBuf>,
    pub no_std: bool,
    pub codegen_backends: Option<Vec<Interned<String>>>,
}

impl Target {
    pub fn from_triple(triple: &str) -> Self {
        let mut target: Self = Default::default();
        if triple.contains("-none") || triple.contains("nvptx") || triple.contains("switch") {
            target.no_std = true;
        }
        target
    }
}
/// Structure of the `config.toml` file that configuration is read from.
///
/// This structure uses `Decodable` to automatically decode a TOML configuration
/// file into this format, and then this is traversed and written into the above
/// `Config` structure.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub(crate) struct TomlConfig {
    changelog_seen: Option<usize>, // FIXME: Deprecated field. Remove it at 2024.
    change_id: Option<usize>,
    build: Option<Build>,
    install: Option<Install>,
    llvm: Option<Llvm>,
    rust: Option<Rust>,
    target: Option<HashMap<String, TomlTarget>>,
    dist: Option<Dist>,
    profile: Option<String>,
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
        TomlConfig {
            build,
            install,
            llvm,
            rust,
            dist,
            target,
            profile: _,
            changelog_seen,
            change_id,
        }: Self,
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
        self.changelog_seen.merge(changelog_seen, replace);
        self.change_id.merge(change_id, replace);
        do_merge(&mut self.build, build, replace);
        do_merge(&mut self.install, install, replace);
        do_merge(&mut self.llvm, llvm, replace);
        do_merge(&mut self.rust, rust, replace);
        do_merge(&mut self.dist, dist, replace);
        assert!(target.is_none(), "merging target-specific config is not currently supported");
    }
}

// We are using a decl macro instead of a derive proc macro here to reduce the compile time of
// rustbuild.
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
        // compile time of rustbuild.
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
        host: Option<Vec<String>> = "host",
        target: Option<Vec<String>> = "target",
        build_dir: Option<String> = "build-dir",
        cargo: Option<String> = "cargo",
        rustc: Option<String> = "rustc",
        rustfmt: Option<PathBuf> = "rustfmt",
        docs: Option<bool> = "docs",
        compiler_docs: Option<bool> = "compiler-docs",
        library_docs_private_items: Option<bool> = "library-docs-private-items",
        docs_minification: Option<bool> = "docs-minification",
        submodules: Option<bool> = "submodules",
        gdb: Option<String> = "gdb",
        nodejs: Option<String> = "nodejs",
        npm: Option<String> = "npm",
        python: Option<String> = "python",
        reuse: Option<String> = "reuse",
        locked_deps: Option<bool> = "locked-deps",
        vendor: Option<bool> = "vendor",
        full_bootstrap: Option<bool> = "full-bootstrap",
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
        plugins: Option<bool> = "plugins",
        ccache: Option<StringOrBool> = "ccache",
        static_libstdcpp: Option<bool> = "static-libstdcpp",
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
        polly: Option<bool> = "polly",
        clang: Option<bool> = "clang",
        enable_warnings: Option<bool> = "enable-warnings",
        download_ci_llvm: Option<StringOrBool> = "download-ci-llvm",
        build_config: Option<HashMap<String, String>> = "build-config",
    }
}

define_config! {
    struct Dist {
        sign_folder: Option<String> = "sign-folder",
        upload_addr: Option<String> = "upload-addr",
        src_tarball: Option<bool> = "src-tarball",
        missing_tools: Option<bool> = "missing-tools",
        compression_formats: Option<Vec<String>> = "compression-formats",
        compression_profile: Option<String> = "compression-profile",
        include_mingw_linker: Option<bool> = "include-mingw-linker",
    }
}

#[derive(Clone, Debug, Deserialize)]
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

impl<'de> serde::de::Visitor<'de> for OptimizeVisitor {
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
enum StringOrInt<'a> {
    String(&'a str),
    Int(i64),
}

impl<'de> Deserialize<'de> for LldMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LldModeVisitor;

        impl<'de> serde::de::Visitor<'de> for LldModeVisitor {
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
                    _ => Err(E::custom("unknown mode {v}")),
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
        debug_assertions: Option<bool> = "debug-assertions",
        debug_assertions_std: Option<bool> = "debug-assertions-std",
        overflow_checks: Option<bool> = "overflow-checks",
        overflow_checks_std: Option<bool> = "overflow-checks-std",
        debug_logging: Option<bool> = "debug-logging",
        debuginfo_level: Option<DebuginfoLevel> = "debuginfo-level",
        debuginfo_level_rustc: Option<DebuginfoLevel> = "debuginfo-level-rustc",
        debuginfo_level_std: Option<DebuginfoLevel> = "debuginfo-level-std",
        debuginfo_level_tools: Option<DebuginfoLevel> = "debuginfo-level-tools",
        debuginfo_level_tests: Option<DebuginfoLevel> = "debuginfo-level-tests",
        split_debuginfo: Option<String> = "split-debuginfo",
        backtrace: Option<bool> = "backtrace",
        incremental: Option<bool> = "incremental",
        parallel_compiler: Option<bool> = "parallel-compiler",
        default_linker: Option<String> = "default-linker",
        channel: Option<String> = "channel",
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
    }
}

impl Config {
    pub fn default_opts() -> Config {
        Config {
            bypass_bootstrap_lock: false,
            llvm_optimize: true,
            ninja_in_file: true,
            llvm_static_stdcpp: false,
            backtrace: true,
            rust_optimize: RustOptimize::Bool(true),
            rust_optimize_tests: true,
            submodules: None,
            docs: true,
            docs_minification: true,
            rust_rpath: true,
            rust_strip: false,
            channel: "dev".to_string(),
            codegen_tests: true,
            rust_dist_src: true,
            rust_codegen_backends: vec![INTERNER.intern_str("llvm")],
            deny_warnings: true,
            bindir: "bin".into(),
            dist_include_mingw_linker: true,
            dist_compression_profile: "fast".into(),
            rustc_parallel: true,

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

            ..Default::default()
        }
    }

    pub fn parse(args: &[String]) -> Config {
        #[cfg(test)]
        fn get_toml(_: &Path) -> TomlConfig {
            TomlConfig::default()
        }

        #[cfg(not(test))]
        fn get_toml(file: &Path) -> TomlConfig {
            let contents =
                t!(fs::read_to_string(file), format!("config file {} not found", file.display()));
            // Deserialize to Value and then TomlConfig to prevent the Deserialize impl of
            // TomlConfig and sub types to be monomorphized 5x by toml.
            toml::from_str(&contents)
                .and_then(|table: toml::Value| TomlConfig::deserialize(table))
                .unwrap_or_else(|err| {
                    eprintln!("failed to parse TOML configuration '{}': {err}", file.display());
                    exit!(2);
                })
        }
        Self::parse_inner(args, get_toml)
    }

    pub(crate) fn parse_inner(args: &[String], get_toml: impl Fn(&Path) -> TomlConfig) -> Config {
        let mut flags = Flags::parse(args);
        let mut config = Config::default_opts();

        // Set flags.
        config.paths = std::mem::take(&mut flags.paths);
        config.skip = flags.skip.into_iter().chain(flags.exclude).collect();
        config.include_default_paths = flags.include_default_paths;
        config.rustc_error_format = flags.rustc_error_format;
        config.json_output = flags.json_output;
        config.on_fail = flags.on_fail;
        config.jobs = Some(threads_from_config(flags.jobs as u32));
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

        // Infer the rest of the configuration.

        // Infer the source directory. This is non-trivial because we want to support a downloaded bootstrap binary,
        // running on a completely machine from where it was compiled.
        let mut cmd = Command::new("git");
        // NOTE: we cannot support running from outside the repository because the only path we have available
        // is set at compile time, which can be wrong if bootstrap was downloaded from source.
        // We still support running outside the repository if we find we aren't in a git directory.
        cmd.arg("rev-parse").arg("--show-toplevel");
        // Discard stderr because we expect this to fail when building from a tarball.
        let output = cmd
            .stderr(std::process::Stdio::null())
            .output()
            .ok()
            .and_then(|output| if output.status.success() { Some(output) } else { None });
        if let Some(output) = output {
            let git_root = String::from_utf8(output.stdout).unwrap();
            // We need to canonicalize this path to make sure it uses backslashes instead of forward slashes.
            let git_root = PathBuf::from(git_root.trim()).canonicalize().unwrap();
            let s = git_root.to_str().unwrap();

            // Bootstrap is quite bad at handling /? in front of paths
            let src = match s.strip_prefix("\\\\?\\") {
                Some(p) => PathBuf::from(p),
                None => git_root,
            };
            // If this doesn't have at least `stage0.json`, we guessed wrong. This can happen when,
            // for example, the build directory is inside of another unrelated git directory.
            // In that case keep the original `CARGO_MANIFEST_DIR` handling.
            //
            // NOTE: this implies that downloadable bootstrap isn't supported when the build directory is outside
            // the source directory. We could fix that by setting a variable from all three of python, ./x, and x.ps1.
            if src.join("src").join("stage0.json").exists() {
                config.src = src;
            }
        } else {
            // We're building from a tarball, not git sources.
            // We don't support pre-downloaded bootstrap in this case.
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

        let stage0_json = t!(std::fs::read(config.src.join("src").join("stage0.json")));

        config.stage0_metadata = t!(serde_json::from_slice::<Stage0Metadata>(&stage0_json));

        // Read from `--config`, then `RUST_BOOTSTRAP_CONFIG`, then `./config.toml`, then `config.toml` in the root directory.
        let toml_path = flags
            .config
            .clone()
            .or_else(|| env::var_os("RUST_BOOTSTRAP_CONFIG").map(PathBuf::from));
        let using_default_path = toml_path.is_none();
        let mut toml_path = toml_path.unwrap_or_else(|| PathBuf::from("config.toml"));
        if using_default_path && !toml_path.exists() {
            toml_path = config.src.join(toml_path);
        }

        // Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
        // but not if `config.toml` hasn't been created.
        let mut toml = if !using_default_path || toml_path.exists() {
            config.config = Some(toml_path.clone());
            get_toml(&toml_path)
        } else {
            config.config = None;
            TomlConfig::default()
        };

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
            include_path.push(format!("config.{include}.toml"));
            let included_toml = get_toml(&include_path);
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

        config.changelog_seen = toml.changelog_seen;
        config.change_id = toml.change_id;

        let Build {
            build,
            host,
            target,
            build_dir,
            cargo,
            rustc,
            rustfmt,
            docs,
            compiler_docs,
            library_docs_private_items,
            docs_minification,
            submodules,
            gdb,
            nodejs,
            npm,
            python,
            reuse,
            locked_deps,
            vendor,
            full_bootstrap,
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
        } = toml.build.unwrap_or_default();

        if let Some(file_build) = build {
            config.build = TargetSelection::from_user(&file_build);
        };

        set(&mut config.out, flags.build_dir.or_else(|| build_dir.map(PathBuf::from)));
        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        if !config.out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            config.out = crate::utils::helpers::absolute(&config.out);
        }

        config.initial_rustc = if let Some(rustc) = rustc {
            if !flags.skip_stage0_validation {
                config.check_build_rustc_version(&rustc);
            }
            PathBuf::from(rustc)
        } else {
            config.download_beta_toolchain();
            config.out.join(config.build.triple).join("stage0/bin/rustc")
        };

        config.initial_cargo = cargo
            .map(|cargo| {
                t!(PathBuf::from(cargo).canonicalize(), "`initial_cargo` not found on disk")
            })
            .unwrap_or_else(|| config.out.join(config.build.triple).join("stage0/bin/cargo"));

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
        config.python = python.map(PathBuf::from);
        config.reuse = reuse.map(PathBuf::from);
        config.submodules = submodules;
        config.android_ndk = android_ndk;
        set(&mut config.low_priority, low_priority);
        set(&mut config.compiler_docs, compiler_docs);
        set(&mut config.library_docs_private_items, library_docs_private_items);
        set(&mut config.docs_minification, docs_minification);
        set(&mut config.docs, docs);
        set(&mut config.locked_deps, locked_deps);
        set(&mut config.vendor, vendor);
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

        // Store off these values as options because if they're not provided
        // we'll infer default values for them later
        let mut llvm_assertions = None;
        let mut llvm_tests = None;
        let mut llvm_plugins = None;
        let mut debug = None;
        let mut debug_assertions = None;
        let mut debug_assertions_std = None;
        let mut overflow_checks = None;
        let mut overflow_checks_std = None;
        let mut debug_logging = None;
        let mut debuginfo_level = None;
        let mut debuginfo_level_rustc = None;
        let mut debuginfo_level_std = None;
        let mut debuginfo_level_tools = None;
        let mut debuginfo_level_tests = None;
        let mut optimize = None;
        let mut omit_git_hash = None;

        if let Some(rust) = toml.rust {
            let Rust {
                optimize: optimize_toml,
                debug: debug_toml,
                codegen_units,
                codegen_units_std,
                debug_assertions: debug_assertions_toml,
                debug_assertions_std: debug_assertions_std_toml,
                overflow_checks: overflow_checks_toml,
                overflow_checks_std: overflow_checks_std_toml,
                debug_logging: debug_logging_toml,
                debuginfo_level: debuginfo_level_toml,
                debuginfo_level_rustc: debuginfo_level_rustc_toml,
                debuginfo_level_std: debuginfo_level_std_toml,
                debuginfo_level_tools: debuginfo_level_tools_toml,
                debuginfo_level_tests: debuginfo_level_tests_toml,
                split_debuginfo,
                backtrace,
                incremental,
                parallel_compiler,
                default_linker,
                channel,
                description,
                musl_root,
                rpath,
                verbose_tests,
                optimize_tests,
                codegen_tests,
                omit_git_hash: omit_git_hash_toml,
                dist_src,
                save_toolstates,
                codegen_backends,
                lld,
                llvm_tools,
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
            } = rust;

            set(&mut config.channel, channel);

            config.download_rustc_commit = config.download_ci_rustc_commit(download_rustc);
            // This list is incomplete, please help by expanding it!
            if config.download_rustc_commit.is_some() {
                // We need the channel used by the downloaded compiler to match the one we set for rustdoc;
                // otherwise rustdoc-ui tests break.
                let ci_channel = t!(fs::read_to_string(config.src.join("src/ci/channel")));
                let ci_channel = ci_channel.trim_end();
                if config.channel != ci_channel
                    && !(config.channel == "dev" && ci_channel == "nightly")
                {
                    panic!(
                        "setting rust.channel={} is incompatible with download-rustc",
                        config.channel
                    );
                }
            }

            debug = debug_toml;
            debug_assertions = debug_assertions_toml;
            debug_assertions_std = debug_assertions_std_toml;
            overflow_checks = overflow_checks_toml;
            overflow_checks_std = overflow_checks_std_toml;
            debug_logging = debug_logging_toml;
            debuginfo_level = debuginfo_level_toml;
            debuginfo_level_rustc = debuginfo_level_rustc_toml;
            debuginfo_level_std = debuginfo_level_std_toml;
            debuginfo_level_tools = debuginfo_level_tools_toml;
            debuginfo_level_tests = debuginfo_level_tests_toml;

            config.rust_split_debuginfo = split_debuginfo
                .as_deref()
                .map(SplitDebuginfo::from_str)
                .map(|v| v.expect("invalid value for rust.split_debuginfo"))
                .unwrap_or(SplitDebuginfo::default_for_platform(config.build));
            optimize = optimize_toml;
            omit_git_hash = omit_git_hash_toml;
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
            config.description = description;
            set(&mut config.rust_dist_src, dist_src);
            set(&mut config.verbose_tests, verbose_tests);
            // in the case "false" is set explicitly, do not overwrite the command line args
            if let Some(true) = incremental {
                config.incremental = true;
            }
            set(&mut config.lld_mode, lld_mode);
            set(&mut config.lld_enabled, lld);

            if matches!(config.lld_mode, LldMode::SelfContained)
                && !config.lld_enabled
                && flags.stage.unwrap_or(0) > 0
            {
                panic!(
                    "Trying to use self-contained lld as a linker, but LLD is not being added to the sysroot. Enable it with rust.lld = true."
                );
            }

            config.llvm_tools_enabled = llvm_tools.unwrap_or(true);
            config.rustc_parallel =
                parallel_compiler.unwrap_or(config.channel == "dev" || config.channel == "nightly");
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

                    INTERNER.intern_str(s)
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

        // rust_info must be set before is_ci_llvm_available() is called.
        let default = config.channel == "dev";
        config.omit_git_hash = omit_git_hash.unwrap_or(default);
        config.rust_info = GitInfo::new(config.omit_git_hash, &config.src);

        if let Some(llvm) = toml.llvm {
            let Llvm {
                optimize: optimize_toml,
                thin_lto,
                release_debuginfo,
                assertions,
                tests,
                plugins,
                ccache,
                static_libstdcpp,
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
                polly,
                clang,
                enable_warnings,
                download_ci_llvm,
                build_config,
            } = llvm;
            match ccache {
                Some(StringOrBool::String(ref s)) => config.ccache = Some(s.to_string()),
                Some(StringOrBool::Bool(true)) => {
                    config.ccache = Some("ccache".to_string());
                }
                Some(StringOrBool::Bool(false)) | None => {}
            }
            set(&mut config.ninja_in_file, ninja);
            llvm_assertions = assertions;
            llvm_tests = tests;
            llvm_plugins = plugins;
            set(&mut config.llvm_optimize, optimize_toml);
            set(&mut config.llvm_thin_lto, thin_lto);
            set(&mut config.llvm_release_debuginfo, release_debuginfo);
            set(&mut config.llvm_static_stdcpp, static_libstdcpp);
            if let Some(v) = link_shared {
                config.llvm_link_shared.set(Some(v));
            }
            config.llvm_targets = targets.clone();
            config.llvm_experimental_targets = experimental_targets.clone();
            config.llvm_link_jobs = link_jobs;
            config.llvm_version_suffix = version_suffix.clone();
            config.llvm_clang_cl = clang_cl.clone();

            config.llvm_cflags = cflags.clone();
            config.llvm_cxxflags = cxxflags.clone();
            config.llvm_ldflags = ldflags.clone();
            set(&mut config.llvm_use_libcxx, use_libcxx);
            config.llvm_use_linker = use_linker.clone();
            config.llvm_allow_old_toolchain = allow_old_toolchain.unwrap_or(false);
            config.llvm_polly = polly.unwrap_or(false);
            config.llvm_clang = clang.unwrap_or(false);
            config.llvm_enable_warnings = enable_warnings.unwrap_or(false);
            config.llvm_build_config = build_config.clone().unwrap_or(Default::default());

            let asserts = llvm_assertions.unwrap_or(false);
            config.llvm_from_ci = config.parse_download_ci_llvm(download_ci_llvm, asserts);

            if config.llvm_from_ci {
                // None of the LLVM options, except assertions, are supported
                // when using downloaded LLVM. We could just ignore these but
                // that's potentially confusing, so force them to not be
                // explicitly set. The defaults and CI defaults don't
                // necessarily match but forcing people to match (somewhat
                // arbitrary) CI configuration locally seems bad/hard.
                check_ci_llvm!(optimize_toml);
                check_ci_llvm!(thin_lto);
                check_ci_llvm!(release_debuginfo);
                // CI-built LLVM can be either dynamic or static. We won't know until we download it.
                check_ci_llvm!(link_shared);
                check_ci_llvm!(static_libstdcpp);
                check_ci_llvm!(targets);
                check_ci_llvm!(experimental_targets);
                check_ci_llvm!(clang_cl);
                check_ci_llvm!(version_suffix);
                check_ci_llvm!(cflags);
                check_ci_llvm!(cxxflags);
                check_ci_llvm!(ldflags);
                check_ci_llvm!(use_libcxx);
                check_ci_llvm!(use_linker);
                check_ci_llvm!(allow_old_toolchain);
                check_ci_llvm!(polly);
                check_ci_llvm!(clang);
                check_ci_llvm!(build_config);
                check_ci_llvm!(plugins);
            }

            // NOTE: can never be hit when downloading from CI, since we call `check_ci_llvm!(thin_lto)` above.
            if config.llvm_thin_lto && link_shared.is_none() {
                // If we're building with ThinLTO on, by default we want to link
                // to LLVM shared, to avoid re-doing ThinLTO (which happens in
                // the link step) with each stage.
                config.llvm_link_shared.set(Some(true));
            }
        } else {
            config.llvm_from_ci = config.parse_download_ci_llvm(None, false);
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
                target.llvm_libunwind = cfg
                    .llvm_libunwind
                    .as_ref()
                    .map(|v| v.parse().expect("failed to parse rust.llvm-libunwind"));
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
                target.sanitizers = cfg.sanitizers;
                target.profiler = cfg.profiler;
                target.rpath = cfg.rpath;

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

                        INTERNER.intern_str(s)
                    }).collect());
                }

                config.target_config.insert(TargetSelection::from_user(&triple), target);
            }
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
                missing_tools,
                compression_formats,
                compression_profile,
                include_mingw_linker,
            } = dist;
            config.dist_sign_folder = sign_folder.map(PathBuf::from);
            config.dist_upload_addr = upload_addr;
            config.dist_compression_formats = compression_formats;
            set(&mut config.dist_compression_profile, compression_profile);
            set(&mut config.rust_dist_src, src_tarball);
            set(&mut config.missing_tools, missing_tools);
            set(&mut config.dist_include_mingw_linker, include_mingw_linker)
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

        config.llvm_assertions = llvm_assertions.unwrap_or(false);
        config.llvm_tests = llvm_tests.unwrap_or(false);
        config.llvm_plugins = llvm_plugins.unwrap_or(false);
        config.rust_optimize = optimize.unwrap_or(RustOptimize::Bool(true));

        let default = debug == Some(true);
        config.rust_debug_assertions = debug_assertions.unwrap_or(default);
        config.rust_debug_assertions_std =
            debug_assertions_std.unwrap_or(config.rust_debug_assertions);
        config.rust_overflow_checks = overflow_checks.unwrap_or(default);
        config.rust_overflow_checks_std =
            overflow_checks_std.unwrap_or(config.rust_overflow_checks);

        config.rust_debug_logging = debug_logging.unwrap_or(config.rust_debug_assertions);

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

        let download_rustc = config.download_rustc_commit.is_some();
        // See https://github.com/rust-lang/compiler-team/issues/326
        config.stage = match config.cmd {
            Subcommand::Check { .. } => flags.stage.or(check_stage).unwrap_or(0),
            // `download-rustc` only has a speed-up for stage2 builds. Default to stage2 unless explicitly overridden.
            Subcommand::Doc { .. } => {
                flags.stage.or(doc_stage).unwrap_or(if download_rustc { 2 } else { 0 })
            }
            Subcommand::Build { .. } => {
                flags.stage.or(build_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Test { .. } => {
                flags.stage.or(test_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Bench { .. } => flags.stage.or(bench_stage).unwrap_or(2),
            Subcommand::Dist { .. } => flags.stage.or(dist_stage).unwrap_or(2),
            Subcommand::Install { .. } => flags.stage.or(install_stage).unwrap_or(2),
            // These are all bootstrap tools, which don't depend on the compiler.
            // The stage we pass shouldn't matter, but use 0 just in case.
            Subcommand::Clean { .. }
            | Subcommand::Clippy { .. }
            | Subcommand::Fix { .. }
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. }
            | Subcommand::Suggest { .. } => flags.stage.unwrap_or(0),
        };

        // CI should always run stage 2 builds, unless it specifically states otherwise
        #[cfg(not(test))]
        if flags.stage.is_none() && crate::CiEnv::current() != crate::CiEnv::None {
            match config.cmd {
                Subcommand::Test { .. }
                | Subcommand::Doc { .. }
                | Subcommand::Build { .. }
                | Subcommand::Bench { .. }
                | Subcommand::Dist { .. }
                | Subcommand::Install { .. } => {
                    assert_eq!(
                        config.stage, 2,
                        "x.py should be run with `--stage 2` on CI, but was run with `--stage {}`",
                        config.stage,
                    );
                }
                Subcommand::Clean { .. }
                | Subcommand::Check { .. }
                | Subcommand::Clippy { .. }
                | Subcommand::Fix { .. }
                | Subcommand::Run { .. }
                | Subcommand::Setup { .. }
                | Subcommand::Format { .. }
                | Subcommand::Suggest { .. } => {}
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

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    #[deprecated = "use `Builder::try_run` instead where possible"]
    pub(crate) fn try_run(&self, cmd: &mut Command) -> Result<(), ()> {
        if self.dry_run() {
            return Ok(());
        }
        self.verbose(&format!("running: {cmd:?}"));
        build_helper::util::try_run(cmd, self.is_verbose())
    }

    /// A git invocation which runs inside the source directory.
    ///
    /// Use this rather than `Command::new("git")` in order to support out-of-tree builds.
    pub(crate) fn git(&self) -> Command {
        let mut git = Command::new("git");
        git.current_dir(&self.src);
        git
    }

    pub(crate) fn test_args(&self) -> Vec<&str> {
        let mut test_args = match self.cmd {
            Subcommand::Test { ref test_args, .. } | Subcommand::Bench { ref test_args, .. } => {
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

    /// Bootstrap embeds a version number into the name of shared libraries it uploads in CI.
    /// Return the version it would have used for the given commit.
    pub(crate) fn artifact_version_part(&self, commit: &str) -> String {
        let (channel, version) = if self.rust_info.is_managed_git_subrepository() {
            let mut channel = self.git();
            channel.arg("show").arg(format!("{commit}:src/ci/channel"));
            let channel = output(&mut channel);
            let mut version = self.git();
            version.arg("show").arg(format!("{commit}:src/version"));
            let version = output(&mut version);
            (channel.trim().to_owned(), version.trim().to_owned())
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
        self.out.join(&*self.build.triple).join("ci-llvm")
    }

    /// Directory where the extracted `rustc-dev` component is stored.
    pub(crate) fn ci_rustc_dir(&self) -> PathBuf {
        assert!(self.download_rustc());
        self.out.join(self.build.triple).join("ci-rustc")
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

    pub fn verbose(&self, msg: &str) {
        if self.verbose > 0 {
            println!("{msg}");
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

    pub fn llvm_enabled(&self, target: TargetSelection) -> bool {
        self.codegen_backends(target).contains(&INTERNER.intern_str("llvm"))
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

    pub fn submodules(&self, rust_info: &GitInfo) -> bool {
        self.submodules.unwrap_or(rust_info.is_managed_git_subrepository())
    }

    pub fn codegen_backends(&self, target: TargetSelection) -> &[Interned<String>] {
        self.target_config
            .get(&target)
            .and_then(|cfg| cfg.codegen_backends.as_deref())
            .unwrap_or(&self.rust_codegen_backends)
    }

    pub fn default_codegen_backend(&self, target: TargetSelection) -> Option<Interned<String>> {
        self.codegen_backends(target).first().cloned()
    }

    pub fn git_config(&self) -> GitConfig<'_> {
        GitConfig {
            git_repository: &self.stage0_metadata.config.git_repository,
            nightly_branch: &self.stage0_metadata.config.nightly_branch,
        }
    }

    pub fn check_build_rustc_version(&self, rustc_path: &str) {
        if self.dry_run() {
            return;
        }

        // check rustc version is same or lower with 1 apart from the building one
        let mut cmd = Command::new(rustc_path);
        cmd.arg("--version");
        let rustc_output = output(&mut cmd)
            .lines()
            .next()
            .unwrap()
            .split(' ')
            .nth(1)
            .unwrap()
            .split('-')
            .next()
            .unwrap()
            .to_owned();
        let rustc_version = Version::parse(rustc_output.trim()).unwrap();
        let source_version =
            Version::parse(fs::read_to_string(self.src.join("src/version")).unwrap().trim())
                .unwrap();
        if !(source_version == rustc_version
            || (source_version.major == rustc_version.major
                && (source_version.minor == rustc_version.minor
                    || source_version.minor == rustc_version.minor + 1)))
        {
            let prev_version = format!("{}.{}.x", source_version.major, source_version.minor - 1);
            eprintln!(
                "Unexpected rustc version: {rustc_version}, we should use {prev_version}/{source_version} to build source with {source_version}"
            );
            exit!(1);
        }
    }

    /// Returns the commit to download, or `None` if we shouldn't download CI artifacts.
    fn download_ci_rustc_commit(&self, download_rustc: Option<StringOrBool>) -> Option<String> {
        // If `download-rustc` is not set, default to rebuilding.
        let if_unchanged = match download_rustc {
            None | Some(StringOrBool::Bool(false)) => return None,
            Some(StringOrBool::Bool(true)) => false,
            Some(StringOrBool::String(s)) if s == "if-unchanged" => true,
            Some(StringOrBool::String(other)) => {
                panic!("unrecognized option for download-rustc: {other}")
            }
        };

        // Handle running from a directory other than the top level
        let top_level = output(self.git().args(["rev-parse", "--show-toplevel"]));
        let top_level = top_level.trim_end();
        let compiler = format!("{top_level}/compiler/");
        let library = format!("{top_level}/library/");

        // Look for a version to compare to based on the current commit.
        // Only commits merged by bors will have CI artifacts.
        let merge_base = output(
            self.git()
                .arg("rev-list")
                .arg(format!("--author={}", self.stage0_metadata.config.git_merge_commit_email))
                .args(["-n1", "--first-parent", "HEAD"]),
        );
        let commit = merge_base.trim_end();
        if commit.is_empty() {
            println!("ERROR: could not find commit hash for downloading rustc");
            println!("HELP: maybe your repository history is too shallow?");
            println!("HELP: consider disabling `download-rustc`");
            println!("HELP: or fetch enough history to include one upstream commit");
            crate::exit!(1);
        }

        // Warn if there were changes to the compiler or standard library since the ancestor commit.
        let has_changes = !t!(self
            .git()
            .args(["diff-index", "--quiet", commit, "--", &compiler, &library])
            .status())
        .success();
        if has_changes {
            if if_unchanged {
                if self.verbose > 0 {
                    println!(
                        "WARNING: saw changes to compiler/ or library/ since {commit}; \
                            ignoring `download-rustc`"
                    );
                }
                return None;
            }
            println!(
                "WARNING: `download-rustc` is enabled, but there are changes to \
                    compiler/ or library/"
            );
        }

        Some(commit.to_string())
    }

    fn parse_download_ci_llvm(
        &self,
        download_ci_llvm: Option<StringOrBool>,
        asserts: bool,
    ) -> bool {
        let if_unchanged = || {
            // Git is needed to track modifications here, but tarball source is not available.
            // If not modified here or built through tarball source, we maintain consistency
            // with '"if available"'.
            if !self.rust_info.is_from_tarball()
                && self
                    .last_modified_commit(&["src/llvm-project"], "download-ci-llvm", true)
                    .is_none()
            {
                // there are some untracked changes in the the given paths.
                false
            } else {
                llvm::is_ci_llvm_available(self, asserts)
            }
        };
        match download_ci_llvm {
            None => self.channel == "dev" && if_unchanged(),
            Some(StringOrBool::Bool(b)) => b,
            // FIXME: "if-available" is deprecated. Remove this block later (around mid 2024)
            // to not break builds between the recent-to-old checkouts.
            Some(StringOrBool::String(s)) if s == "if-available" => {
                llvm::is_ci_llvm_available(self, asserts)
            }
            Some(StringOrBool::String(s)) if s == "if-unchanged" => if_unchanged(),
            Some(StringOrBool::String(other)) => {
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
        // Handle running from a directory other than the top level
        let top_level = output(self.git().args(["rev-parse", "--show-toplevel"]));
        let top_level = top_level.trim_end();

        // Look for a version to compare to based on the current commit.
        // Only commits merged by bors will have CI artifacts.
        let merge_base = output(
            self.git()
                .arg("rev-list")
                .arg(format!("--author={}", self.stage0_metadata.config.git_merge_commit_email))
                .args(["-n1", "--first-parent", "HEAD"]),
        );
        let commit = merge_base.trim_end();
        if commit.is_empty() {
            println!("error: could not find commit hash for downloading components from CI");
            println!("help: maybe your repository history is too shallow?");
            println!("help: consider disabling `{option_name}`");
            println!("help: or fetch enough history to include one upstream commit");
            crate::exit!(1);
        }

        // Warn if there were changes to the compiler or standard library since the ancestor commit.
        let mut git = self.git();
        git.args(["diff-index", "--quiet", commit, "--"]);

        for path in modified_paths {
            git.arg(format!("{top_level}/{path}"));
        }

        let has_changes = !t!(git.status()).success();
        if has_changes {
            if if_unchanged {
                if self.verbose > 0 {
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
