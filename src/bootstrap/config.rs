//! Serialized configuration of a build.
//!
//! This module implements parsing `config.toml` configuration files to tweak
//! how the build runs.

#[cfg(test)]
mod tests;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use crate::builder::TaskPath;
use crate::cache::{Interned, INTERNER};
use crate::cc_detect::{ndk_compiler, Language};
use crate::channel::{self, GitInfo};
pub use crate::flags::Subcommand;
use crate::flags::{Color, Flags};
use crate::util::{exe, output, t};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Deserializer};

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

/// Global configuration for the entire build and/or bootstrap.
///
/// This structure is parsed from `config.toml`, and some of the fields are inferred from `git` or build-time parameters.
///
/// Note that this structure is not decoded directly into, but rather it is
/// filled out from the decoded forms of the structs below. For documentation
/// each field, see the corresponding fields in
/// `config.toml.example`.
#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
pub struct Config {
    pub changelog_seen: Option<usize>,
    pub ccache: Option<String>,
    /// Call Build::ninja() instead of this.
    pub ninja_in_file: bool,
    pub verbose: usize,
    pub submodules: Option<bool>,
    pub compiler_docs: bool,
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
    pub ignore_git: bool,
    pub exclude: Vec<TaskPath>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
    pub test_compare_mode: bool,
    pub color: Color,
    pub patch_binaries_for_nix: bool,
    pub stage0_metadata: Stage0Metadata,

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
    /// `None` if we shouldn't download CI compiler artifacts, or the commit to download if we should.
    #[cfg(not(test))]
    download_rustc_commit: Option<String>,
    #[cfg(test)]
    pub download_rustc_commit: Option<String>,

    pub deny_warnings: bool,
    pub backtrace_on_ice: bool,

    // llvm codegen options
    pub llvm_skip_rebuild: bool,
    pub llvm_assertions: bool,
    pub llvm_tests: bool,
    pub llvm_plugins: bool,
    pub llvm_optimize: bool,
    pub llvm_thin_lto: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_version_check: bool,
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
    pub llvm_from_ci: bool,
    pub llvm_build_config: HashMap<String, String>,

    pub use_lld: bool,
    pub lld_enabled: bool,
    pub llvm_tools_enabled: bool,

    pub llvm_cflags: Option<String>,
    pub llvm_cxxflags: Option<String>,
    pub llvm_ldflags: Option<String>,
    pub llvm_use_libcxx: bool,

    // rust codegen options
    pub rust_optimize: bool,
    pub rust_codegen_units: Option<u32>,
    pub rust_codegen_units_std: Option<u32>,
    pub rust_debug_assertions: bool,
    pub rust_debug_assertions_std: bool,
    pub rust_overflow_checks: bool,
    pub rust_overflow_checks_std: bool,
    pub rust_debug_logging: bool,
    pub rust_debuginfo_level_rustc: u32,
    pub rust_debuginfo_level_std: u32,
    pub rust_debuginfo_level_tools: u32,
    pub rust_debuginfo_level_tests: u32,
    pub rust_split_debuginfo: SplitDebuginfo,
    pub rust_rpath: bool,
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
    pub llvm_profile_use: Option<String>,
    pub llvm_profile_generate: bool,
    pub llvm_libunwind_default: Option<LlvmLibunwind>,
    pub llvm_bolt_profile_generate: bool,
    pub llvm_bolt_profile_use: Option<String>,

    pub build: TargetSelection,
    pub hosts: Vec<TargetSelection>,
    pub targets: Vec<TargetSelection>,
    pub local_rebuild: bool,
    pub jemalloc: bool,
    pub control_flow_guard: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_compression_formats: Option<Vec<String>>,

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
    pub missing_tools: bool,

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

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,
    #[cfg(not(test))]
    initial_rustfmt: RefCell<RustfmtState>,
    #[cfg(test)]
    pub initial_rustfmt: RefCell<RustfmtState>,
    pub out: PathBuf,
    pub rust_info: channel::GitInfo,
}

#[derive(Default, Deserialize)]
#[cfg_attr(test, derive(Clone))]
pub struct Stage0Metadata {
    pub config: Stage0Config,
    pub checksums_sha256: HashMap<String, String>,
    pub rustfmt: Option<RustfmtMetadata>,
}
#[derive(Default, Deserialize)]
#[cfg_attr(test, derive(Clone))]
pub struct Stage0Config {
    pub dist_server: String,
    pub artifacts_server: String,
    pub artifacts_with_llvm_assertions_server: String,
    pub git_merge_commit_email: String,
    pub nightly_branch: String,
}
#[derive(Default, Deserialize)]
#[cfg_attr(test, derive(Clone))]
pub struct RustfmtMetadata {
    pub date: String,
    pub version: String,
}

#[derive(Clone, Debug)]
pub enum RustfmtState {
    SystemToolchain(PathBuf),
    Downloaded(PathBuf),
    Unavailable,
    LazyEvaluated,
}

impl Default for RustfmtState {
    fn default() -> Self {
        RustfmtState::LazyEvaluated
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlvmLibunwind {
    No,
    InTree,
    System,
}

impl Default for LlvmLibunwind {
    fn default() -> Self {
        Self::No
    }
}

impl FromStr for LlvmLibunwind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "no" => Ok(Self::No),
            "in-tree" => Ok(Self::InTree),
            "system" => Ok(Self::System),
            invalid => Err(format!("Invalid value '{}' for rust.llvm-libunwind config.", invalid)),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitDebuginfo {
    Packed,
    Unpacked,
    Off,
}

impl Default for SplitDebuginfo {
    fn default() -> Self {
        SplitDebuginfo::Off
    }
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
    /// `rust.split-debuginfo` in `config.toml.example`.
    fn default_for_platform(target: &str) -> Self {
        if target.contains("apple") {
            SplitDebuginfo::Unpacked
        } else if target.contains("windows") {
            SplitDebuginfo::Packed
        } else {
            SplitDebuginfo::Off
        }
    }
}

/// LTO mode used for compiling rustc itself.
#[derive(Default, Clone)]
pub enum RustcLto {
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
            _ => Err(format!("Invalid value for rustc LTO: {}", s)),
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetSelection {
    pub triple: Interned<String>,
    file: Option<Interned<String>>,
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

        Self { triple, file }
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
}

impl fmt::Display for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.triple)?;
        if let Some(file) = self.file {
            write!(f, "({})", file)?;
        }
        Ok(())
    }
}

impl fmt::Debug for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl PartialEq<&str> for TargetSelection {
    fn eq(&self, other: &&str) -> bool {
        self.triple == *other
    }
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
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
    pub ndk: Option<PathBuf>,
    pub sanitizers: Option<bool>,
    pub profiler: Option<bool>,
    pub crt_static: Option<bool>,
    pub musl_root: Option<PathBuf>,
    pub musl_libdir: Option<PathBuf>,
    pub wasi_root: Option<PathBuf>,
    pub qemu_rootfs: Option<PathBuf>,
    pub no_std: bool,
}

impl Target {
    pub fn from_triple(triple: &str) -> Self {
        let mut target: Self = Default::default();
        if triple.contains("-none")
            || triple.contains("nvptx")
            || triple.contains("switch")
            || triple.contains("-uefi")
        {
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
struct TomlConfig {
    changelog_seen: Option<usize>,
    build: Option<Build>,
    install: Option<Install>,
    llvm: Option<Llvm>,
    rust: Option<Rust>,
    target: Option<HashMap<String, TomlTarget>>,
    dist: Option<Dist>,
    profile: Option<String>,
}

trait Merge {
    fn merge(&mut self, other: Self);
}

impl Merge for TomlConfig {
    fn merge(
        &mut self,
        TomlConfig { build, install, llvm, rust, dist, target, profile: _, changelog_seen: _ }: Self,
    ) {
        fn do_merge<T: Merge>(x: &mut Option<T>, y: Option<T>) {
            if let Some(new) = y {
                if let Some(original) = x {
                    original.merge(new);
                } else {
                    *x = Some(new);
                }
            }
        }
        do_merge(&mut self.build, build);
        do_merge(&mut self.install, install);
        do_merge(&mut self.llvm, llvm);
        do_merge(&mut self.rust, rust);
        do_merge(&mut self.dist, dist);
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
            fn merge(&mut self, other: Self) {
                $(
                    if !self.$field.is_some() {
                        self.$field = other.$field;
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
        skip_rebuild: Option<bool> = "skip-rebuild",
        optimize: Option<bool> = "optimize",
        thin_lto: Option<bool> = "thin-lto",
        release_debuginfo: Option<bool> = "release-debuginfo",
        assertions: Option<bool> = "assertions",
        tests: Option<bool> = "tests",
        plugins: Option<bool> = "plugins",
        ccache: Option<StringOrBool> = "ccache",
        version_check: Option<bool> = "version-check",
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
        download_ci_llvm: Option<StringOrBool> = "download-ci-llvm",
        build_config: Option<HashMap<String, String>> = "build-config",
    }
}

define_config! {
    struct Dist {
        sign_folder: Option<String> = "sign-folder",
        gpg_password_file: Option<String> = "gpg-password-file",
        upload_addr: Option<String> = "upload-addr",
        src_tarball: Option<bool> = "src-tarball",
        missing_tools: Option<bool> = "missing-tools",
        compression_formats: Option<Vec<String>> = "compression-formats",
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StringOrBool {
    String(String),
    Bool(bool),
}

impl Default for StringOrBool {
    fn default() -> StringOrBool {
        StringOrBool::Bool(false)
    }
}

define_config! {
    /// TOML representation of how the Rust build is configured.
    struct Rust {
        optimize: Option<bool> = "optimize",
        debug: Option<bool> = "debug",
        codegen_units: Option<u32> = "codegen-units",
        codegen_units_std: Option<u32> = "codegen-units-std",
        debug_assertions: Option<bool> = "debug-assertions",
        debug_assertions_std: Option<bool> = "debug-assertions-std",
        overflow_checks: Option<bool> = "overflow-checks",
        overflow_checks_std: Option<bool> = "overflow-checks-std",
        debug_logging: Option<bool> = "debug-logging",
        debuginfo_level: Option<u32> = "debuginfo-level",
        debuginfo_level_rustc: Option<u32> = "debuginfo-level-rustc",
        debuginfo_level_std: Option<u32> = "debuginfo-level-std",
        debuginfo_level_tools: Option<u32> = "debuginfo-level-tools",
        debuginfo_level_tests: Option<u32> = "debuginfo-level-tests",
        split_debuginfo: Option<String> = "split-debuginfo",
        run_dsymutil: Option<bool> = "run-dsymutil",
        backtrace: Option<bool> = "backtrace",
        incremental: Option<bool> = "incremental",
        parallel_compiler: Option<bool> = "parallel-compiler",
        default_linker: Option<String> = "default-linker",
        channel: Option<String> = "channel",
        description: Option<String> = "description",
        musl_root: Option<String> = "musl-root",
        rpath: Option<bool> = "rpath",
        verbose_tests: Option<bool> = "verbose-tests",
        optimize_tests: Option<bool> = "optimize-tests",
        codegen_tests: Option<bool> = "codegen-tests",
        ignore_git: Option<bool> = "ignore-git",
        dist_src: Option<bool> = "dist-src",
        save_toolstates: Option<String> = "save-toolstates",
        codegen_backends: Option<Vec<String>> = "codegen-backends",
        lld: Option<bool> = "lld",
        use_lld: Option<bool> = "use-lld",
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
        new_symbol_mangling: Option<bool> = "new-symbol-mangling",
        profile_generate: Option<String> = "profile-generate",
        profile_use: Option<String> = "profile-use",
        // ignored; this is set from an env var set by bootstrap.py
        download_rustc: Option<StringOrBool> = "download-rustc",
        lto: Option<String> = "lto",
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
        android_ndk: Option<String> = "android-ndk",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<bool> = "profiler",
        crt_static: Option<bool> = "crt-static",
        musl_root: Option<String> = "musl-root",
        musl_libdir: Option<String> = "musl-libdir",
        wasi_root: Option<String> = "wasi-root",
        qemu_rootfs: Option<String> = "qemu-rootfs",
        no_std: Option<bool> = "no-std",
    }
}

impl Config {
    pub fn default_opts() -> Config {
        let mut config = Config::default();
        config.llvm_optimize = true;
        config.ninja_in_file = true;
        config.llvm_version_check = true;
        config.llvm_static_stdcpp = false;
        config.backtrace = true;
        config.rust_optimize = true;
        config.rust_optimize_tests = true;
        config.submodules = None;
        config.docs = true;
        config.docs_minification = true;
        config.rust_rpath = true;
        config.channel = "dev".to_string();
        config.codegen_tests = true;
        config.rust_dist_src = true;
        config.rust_codegen_backends = vec![INTERNER.intern_str("llvm")];
        config.deny_warnings = true;
        config.bindir = "bin".into();

        // set by build.rs
        config.build = TargetSelection::from_user(&env!("BUILD_TRIPLE"));

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Undo `src/bootstrap`
        config.src = manifest_dir.parent().unwrap().parent().unwrap().to_owned();
        config.out = PathBuf::from("build");

        config
    }

    pub fn parse(args: &[String]) -> Config {
        #[cfg(test)]
        let get_toml = |_: &_| TomlConfig::default();
        #[cfg(not(test))]
        let get_toml = |file: &Path| {
            let contents =
                t!(fs::read_to_string(file), format!("config file {} not found", file.display()));
            // Deserialize to Value and then TomlConfig to prevent the Deserialize impl of
            // TomlConfig and sub types to be monomorphized 5x by toml.
            match toml::from_str(&contents)
                .and_then(|table: toml::Value| TomlConfig::deserialize(table))
            {
                Ok(table) => table,
                Err(err) => {
                    eprintln!("failed to parse TOML configuration '{}': {}", file.display(), err);
                    crate::detail_exit(2);
                }
            }
        };

        Self::parse_inner(args, get_toml)
    }

    fn parse_inner<'a>(args: &[String], get_toml: impl 'a + Fn(&Path) -> TomlConfig) -> Config {
        let flags = Flags::parse(&args);
        let mut config = Config::default_opts();

        // Set flags.
        config.exclude = flags.exclude.into_iter().map(|path| TaskPath::parse(path)).collect();
        config.include_default_paths = flags.include_default_paths;
        config.rustc_error_format = flags.rustc_error_format;
        config.json_output = flags.json_output;
        config.on_fail = flags.on_fail;
        config.jobs = flags.jobs.map(threads_from_config);
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.dry_run = if flags.dry_run { DryRun::UserSelected } else { DryRun::Disabled };
        config.keep_stage = flags.keep_stage;
        config.keep_stage_std = flags.keep_stage_std;
        config.color = flags.color;
        if let Some(value) = flags.deny_warnings {
            config.deny_warnings = value;
        }
        config.llvm_profile_use = flags.llvm_profile_use;
        config.llvm_profile_generate = flags.llvm_profile_generate;
        config.llvm_bolt_profile_generate = flags.llvm_bolt_profile_generate;
        config.llvm_bolt_profile_use = flags.llvm_bolt_profile_use;

        if config.llvm_bolt_profile_generate && config.llvm_bolt_profile_use.is_some() {
            eprintln!(
                "Cannot use both `llvm_bolt_profile_generate` and `llvm_bolt_profile_use` at the same time"
            );
            crate::detail_exit(1);
        }

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
                None => PathBuf::from(git_root),
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

        let stage0_json = t!(std::fs::read(&config.src.join("src").join("stage0.json")));

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
            let mut include_path = config.src.clone();
            include_path.push("src");
            include_path.push("bootstrap");
            include_path.push("defaults");
            include_path.push(format!("config.{}.toml", include));
            let included_toml = get_toml(&include_path);
            toml.merge(included_toml);
        }

        config.changelog_seen = toml.changelog_seen;

        let build = toml.build.unwrap_or_default();
        if let Some(file_build) = build.build {
            config.build = TargetSelection::from_user(&file_build);
        };

        set(&mut config.out, flags.build_dir.or_else(|| build.build_dir.map(PathBuf::from)));
        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        if !config.out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            config.out = crate::util::absolute(&config.out);
        }

        config.initial_rustc = build
            .rustc
            .map(PathBuf::from)
            .unwrap_or_else(|| config.out.join(config.build.triple).join("stage0/bin/rustc"));
        config.initial_cargo = build
            .cargo
            .map(PathBuf::from)
            .unwrap_or_else(|| config.out.join(config.build.triple).join("stage0/bin/cargo"));

        // NOTE: it's important this comes *after* we set `initial_rustc` just above.
        if config.dry_run() {
            let dir = config.out.join("tmp-dry-run");
            t!(fs::create_dir_all(&dir));
            config.out = dir;
        }

        config.hosts = if let Some(arg_host) = flags.host {
            arg_host
        } else if let Some(file_host) = build.host {
            file_host.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            vec![config.build]
        };
        config.targets = if let Some(arg_target) = flags.target {
            arg_target
        } else if let Some(file_target) = build.target {
            file_target.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            // If target is *not* configured, then default to the host
            // toolchains.
            config.hosts.clone()
        };

        config.nodejs = build.nodejs.map(PathBuf::from);
        config.npm = build.npm.map(PathBuf::from);
        config.gdb = build.gdb.map(PathBuf::from);
        config.python = build.python.map(PathBuf::from);
        config.reuse = build.reuse.map(PathBuf::from);
        config.submodules = build.submodules;
        set(&mut config.low_priority, build.low_priority);
        set(&mut config.compiler_docs, build.compiler_docs);
        set(&mut config.docs_minification, build.docs_minification);
        set(&mut config.docs, build.docs);
        set(&mut config.locked_deps, build.locked_deps);
        set(&mut config.vendor, build.vendor);
        set(&mut config.full_bootstrap, build.full_bootstrap);
        set(&mut config.extended, build.extended);
        config.tools = build.tools;
        set(&mut config.verbose, build.verbose);
        set(&mut config.sanitizers, build.sanitizers);
        set(&mut config.profiler, build.profiler);
        set(&mut config.cargo_native_static, build.cargo_native_static);
        set(&mut config.configure_args, build.configure_args);
        set(&mut config.local_rebuild, build.local_rebuild);
        set(&mut config.print_step_timings, build.print_step_timings);
        set(&mut config.print_step_rusage, build.print_step_rusage);
        set(&mut config.patch_binaries_for_nix, build.patch_binaries_for_nix);

        config.verbose = cmp::max(config.verbose, flags.verbose);

        if let Some(install) = toml.install {
            config.prefix = install.prefix.map(PathBuf::from);
            config.sysconfdir = install.sysconfdir.map(PathBuf::from);
            config.datadir = install.datadir.map(PathBuf::from);
            config.docdir = install.docdir.map(PathBuf::from);
            set(&mut config.bindir, install.bindir.map(PathBuf::from));
            config.libdir = install.libdir.map(PathBuf::from);
            config.mandir = install.mandir.map(PathBuf::from);
        }

        // We want the llvm-skip-rebuild flag to take precedence over the
        // skip-rebuild config.toml option so we store it separately
        // so that we can infer the right value
        let mut llvm_skip_rebuild = flags.llvm_skip_rebuild;

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
        let mut ignore_git = None;

        if let Some(rust) = toml.rust {
            debug = rust.debug;
            debug_assertions = rust.debug_assertions;
            debug_assertions_std = rust.debug_assertions_std;
            overflow_checks = rust.overflow_checks;
            overflow_checks_std = rust.overflow_checks_std;
            debug_logging = rust.debug_logging;
            debuginfo_level = rust.debuginfo_level;
            debuginfo_level_rustc = rust.debuginfo_level_rustc;
            debuginfo_level_std = rust.debuginfo_level_std;
            debuginfo_level_tools = rust.debuginfo_level_tools;
            debuginfo_level_tests = rust.debuginfo_level_tests;
            config.rust_split_debuginfo = rust
                .split_debuginfo
                .as_deref()
                .map(SplitDebuginfo::from_str)
                .map(|v| v.expect("invalid value for rust.split_debuginfo"))
                .unwrap_or(SplitDebuginfo::default_for_platform(&config.build.triple));
            optimize = rust.optimize;
            ignore_git = rust.ignore_git;
            config.rust_new_symbol_mangling = rust.new_symbol_mangling;
            set(&mut config.rust_optimize_tests, rust.optimize_tests);
            set(&mut config.codegen_tests, rust.codegen_tests);
            set(&mut config.rust_rpath, rust.rpath);
            set(&mut config.jemalloc, rust.jemalloc);
            set(&mut config.test_compare_mode, rust.test_compare_mode);
            set(&mut config.backtrace, rust.backtrace);
            set(&mut config.channel, rust.channel);
            config.description = rust.description;
            set(&mut config.rust_dist_src, rust.dist_src);
            set(&mut config.verbose_tests, rust.verbose_tests);
            // in the case "false" is set explicitly, do not overwrite the command line args
            if let Some(true) = rust.incremental {
                config.incremental = true;
            }
            set(&mut config.use_lld, rust.use_lld);
            set(&mut config.lld_enabled, rust.lld);
            set(&mut config.llvm_tools_enabled, rust.llvm_tools);
            config.rustc_parallel = rust.parallel_compiler.unwrap_or(false);
            config.rustc_default_linker = rust.default_linker;
            config.musl_root = rust.musl_root.map(PathBuf::from);
            config.save_toolstates = rust.save_toolstates.map(PathBuf::from);
            set(&mut config.deny_warnings, flags.deny_warnings.or(rust.deny_warnings));
            set(&mut config.backtrace_on_ice, rust.backtrace_on_ice);
            set(&mut config.rust_verify_llvm_ir, rust.verify_llvm_ir);
            config.rust_thin_lto_import_instr_limit = rust.thin_lto_import_instr_limit;
            set(&mut config.rust_remap_debuginfo, rust.remap_debuginfo);
            set(&mut config.control_flow_guard, rust.control_flow_guard);
            config.llvm_libunwind_default = rust
                .llvm_libunwind
                .map(|v| v.parse().expect("failed to parse rust.llvm-libunwind"));

            if let Some(ref backends) = rust.codegen_backends {
                config.rust_codegen_backends =
                    backends.iter().map(|s| INTERNER.intern_str(s)).collect();
            }

            config.rust_codegen_units = rust.codegen_units.map(threads_from_config);
            config.rust_codegen_units_std = rust.codegen_units_std.map(threads_from_config);
            config.rust_profile_use = flags.rust_profile_use.or(rust.profile_use);
            config.rust_profile_generate = flags.rust_profile_generate.or(rust.profile_generate);
            config.download_rustc_commit = config.download_ci_rustc_commit(rust.download_rustc);

            config.rust_lto = rust
                .lto
                .as_deref()
                .map(|value| RustcLto::from_str(value).unwrap())
                .unwrap_or_default();
        } else {
            config.rust_profile_use = flags.rust_profile_use;
            config.rust_profile_generate = flags.rust_profile_generate;
        }

        if let Some(llvm) = toml.llvm {
            match llvm.ccache {
                Some(StringOrBool::String(ref s)) => config.ccache = Some(s.to_string()),
                Some(StringOrBool::Bool(true)) => {
                    config.ccache = Some("ccache".to_string());
                }
                Some(StringOrBool::Bool(false)) | None => {}
            }
            set(&mut config.ninja_in_file, llvm.ninja);
            llvm_assertions = llvm.assertions;
            llvm_tests = llvm.tests;
            llvm_plugins = llvm.plugins;
            llvm_skip_rebuild = llvm_skip_rebuild.or(llvm.skip_rebuild);
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_thin_lto, llvm.thin_lto);
            set(&mut config.llvm_release_debuginfo, llvm.release_debuginfo);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
            if let Some(v) = llvm.link_shared {
                config.llvm_link_shared.set(Some(v));
            }
            config.llvm_targets = llvm.targets.clone();
            config.llvm_experimental_targets = llvm.experimental_targets.clone();
            config.llvm_link_jobs = llvm.link_jobs;
            config.llvm_version_suffix = llvm.version_suffix.clone();
            config.llvm_clang_cl = llvm.clang_cl.clone();

            config.llvm_cflags = llvm.cflags.clone();
            config.llvm_cxxflags = llvm.cxxflags.clone();
            config.llvm_ldflags = llvm.ldflags.clone();
            set(&mut config.llvm_use_libcxx, llvm.use_libcxx);
            config.llvm_use_linker = llvm.use_linker.clone();
            config.llvm_allow_old_toolchain = llvm.allow_old_toolchain.unwrap_or(false);
            config.llvm_polly = llvm.polly.unwrap_or(false);
            config.llvm_clang = llvm.clang.unwrap_or(false);
            config.llvm_build_config = llvm.build_config.clone().unwrap_or(Default::default());

            let asserts = llvm_assertions.unwrap_or(false);
            config.llvm_from_ci = match llvm.download_ci_llvm {
                Some(StringOrBool::String(s)) => {
                    assert!(s == "if-available", "unknown option `{}` for download-ci-llvm", s);
                    crate::native::is_ci_llvm_available(&config, asserts)
                }
                Some(StringOrBool::Bool(b)) => b,
                None => {
                    config.channel == "dev" && crate::native::is_ci_llvm_available(&config, asserts)
                }
            };

            if config.llvm_from_ci {
                // None of the LLVM options, except assertions, are supported
                // when using downloaded LLVM. We could just ignore these but
                // that's potentially confusing, so force them to not be
                // explicitly set. The defaults and CI defaults don't
                // necessarily match but forcing people to match (somewhat
                // arbitrary) CI configuration locally seems bad/hard.
                check_ci_llvm!(llvm.optimize);
                check_ci_llvm!(llvm.thin_lto);
                check_ci_llvm!(llvm.release_debuginfo);
                // CI-built LLVM can be either dynamic or static. We won't know until we download it.
                check_ci_llvm!(llvm.link_shared);
                check_ci_llvm!(llvm.static_libstdcpp);
                check_ci_llvm!(llvm.targets);
                check_ci_llvm!(llvm.experimental_targets);
                check_ci_llvm!(llvm.link_jobs);
                check_ci_llvm!(llvm.clang_cl);
                check_ci_llvm!(llvm.version_suffix);
                check_ci_llvm!(llvm.cflags);
                check_ci_llvm!(llvm.cxxflags);
                check_ci_llvm!(llvm.ldflags);
                check_ci_llvm!(llvm.use_libcxx);
                check_ci_llvm!(llvm.use_linker);
                check_ci_llvm!(llvm.allow_old_toolchain);
                check_ci_llvm!(llvm.polly);
                check_ci_llvm!(llvm.clang);
                check_ci_llvm!(llvm.build_config);
                check_ci_llvm!(llvm.plugins);
            }

            // NOTE: can never be hit when downloading from CI, since we call `check_ci_llvm!(thin_lto)` above.
            if config.llvm_thin_lto && llvm.link_shared.is_none() {
                // If we're building with ThinLTO on, by default we want to link
                // to LLVM shared, to avoid re-doing ThinLTO (which happens in
                // the link step) with each stage.
                config.llvm_link_shared.set(Some(true));
            }
        } else {
            config.llvm_from_ci =
                config.channel == "dev" && crate::native::is_ci_llvm_available(&config, false);
        }

        if let Some(t) = toml.target {
            for (triple, cfg) in t {
                let mut target = Target::from_triple(&triple);

                if let Some(ref s) = cfg.llvm_config {
                    target.llvm_config = Some(config.src.join(s));
                }
                target.llvm_has_rust_patches = cfg.llvm_has_rust_patches;
                if let Some(ref s) = cfg.llvm_filecheck {
                    target.llvm_filecheck = Some(config.src.join(s));
                }
                target.llvm_libunwind = cfg
                    .llvm_libunwind
                    .as_ref()
                    .map(|v| v.parse().expect("failed to parse rust.llvm-libunwind"));
                if let Some(ref s) = cfg.android_ndk {
                    target.ndk = Some(config.src.join(s));
                }
                if let Some(s) = cfg.no_std {
                    target.no_std = s;
                }
                target.cc = cfg.cc.map(PathBuf::from).or_else(|| {
                    target.ndk.as_ref().map(|ndk| ndk_compiler(Language::C, &triple, ndk))
                });
                target.cxx = cfg.cxx.map(PathBuf::from).or_else(|| {
                    target.ndk.as_ref().map(|ndk| ndk_compiler(Language::CPlusPlus, &triple, ndk))
                });
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

                config.target_config.insert(TargetSelection::from_user(&triple), target);
            }
        }

        if config.llvm_from_ci {
            let triple = &config.build.triple;
            let ci_llvm_bin = config.ci_llvm_root().join("bin");
            let mut build_target = config
                .target_config
                .entry(config.build)
                .or_insert_with(|| Target::from_triple(&triple));

            check_ci_llvm!(build_target.llvm_config);
            check_ci_llvm!(build_target.llvm_filecheck);
            build_target.llvm_config = Some(ci_llvm_bin.join(exe("llvm-config", config.build)));
            build_target.llvm_filecheck = Some(ci_llvm_bin.join(exe("FileCheck", config.build)));
        }

        if let Some(t) = toml.dist {
            config.dist_sign_folder = t.sign_folder.map(PathBuf::from);
            config.dist_upload_addr = t.upload_addr;
            config.dist_compression_formats = t.compression_formats;
            set(&mut config.rust_dist_src, t.src_tarball);
            set(&mut config.missing_tools, t.missing_tools);
        }

        if let Some(r) = build.rustfmt {
            *config.initial_rustfmt.borrow_mut() = if r.exists() {
                RustfmtState::SystemToolchain(r)
            } else {
                RustfmtState::Unavailable
            };
        } else {
            // If using a system toolchain for bootstrapping, see if that has rustfmt available.
            let host = config.build;
            let rustfmt_path = config.initial_rustc.with_file_name(exe("rustfmt", host));
            let bin_root = config.out.join(host.triple).join("stage0");
            if !rustfmt_path.starts_with(&bin_root) {
                // Using a system-provided toolchain; we shouldn't download rustfmt.
                *config.initial_rustfmt.borrow_mut() = RustfmtState::SystemToolchain(rustfmt_path);
            }
        }

        // Now that we've reached the end of our configuration, infer the
        // default values for all options that we haven't otherwise stored yet.

        config.llvm_skip_rebuild = llvm_skip_rebuild.unwrap_or(false);
        config.llvm_assertions = llvm_assertions.unwrap_or(false);
        config.llvm_tests = llvm_tests.unwrap_or(false);
        config.llvm_plugins = llvm_plugins.unwrap_or(false);
        config.rust_optimize = optimize.unwrap_or(true);

        let default = debug == Some(true);
        config.rust_debug_assertions = debug_assertions.unwrap_or(default);
        config.rust_debug_assertions_std =
            debug_assertions_std.unwrap_or(config.rust_debug_assertions);
        config.rust_overflow_checks = overflow_checks.unwrap_or(default);
        config.rust_overflow_checks_std =
            overflow_checks_std.unwrap_or(config.rust_overflow_checks);

        config.rust_debug_logging = debug_logging.unwrap_or(config.rust_debug_assertions);

        let with_defaults = |debuginfo_level_specific: Option<u32>| {
            debuginfo_level_specific.or(debuginfo_level).unwrap_or(if debug == Some(true) {
                1
            } else {
                0
            })
        };
        config.rust_debuginfo_level_rustc = with_defaults(debuginfo_level_rustc);
        config.rust_debuginfo_level_std = with_defaults(debuginfo_level_std);
        config.rust_debuginfo_level_tools = with_defaults(debuginfo_level_tools);
        config.rust_debuginfo_level_tests = debuginfo_level_tests.unwrap_or(0);

        let default = config.channel == "dev";
        config.ignore_git = ignore_git.unwrap_or(default);
        config.rust_info = GitInfo::new(config.ignore_git, &config.src);

        let download_rustc = config.download_rustc_commit.is_some();
        // See https://github.com/rust-lang/compiler-team/issues/326
        config.stage = match config.cmd {
            Subcommand::Check { .. } => flags.stage.or(build.check_stage).unwrap_or(0),
            // `download-rustc` only has a speed-up for stage2 builds. Default to stage2 unless explicitly overridden.
            Subcommand::Doc { .. } => {
                flags.stage.or(build.doc_stage).unwrap_or(if download_rustc { 2 } else { 0 })
            }
            Subcommand::Build { .. } => {
                flags.stage.or(build.build_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Test { .. } => {
                flags.stage.or(build.test_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Bench { .. } => flags.stage.or(build.bench_stage).unwrap_or(2),
            Subcommand::Dist { .. } => flags.stage.or(build.dist_stage).unwrap_or(2),
            Subcommand::Install { .. } => flags.stage.or(build.install_stage).unwrap_or(2),
            // These are all bootstrap tools, which don't depend on the compiler.
            // The stage we pass shouldn't matter, but use 0 just in case.
            Subcommand::Clean { .. }
            | Subcommand::Clippy { .. }
            | Subcommand::Fix { .. }
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. } => flags.stage.unwrap_or(0),
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
                | Subcommand::Format { .. } => {}
            }
        }

        config
    }

    pub(crate) fn dry_run(&self) -> bool {
        match self.dry_run {
            DryRun::Disabled => false,
            DryRun::SelfCheck | DryRun::UserSelected => true,
        }
    }

    /// A git invocation which runs inside the source directory.
    ///
    /// Use this rather than `Command::new("git")` in order to support out-of-tree builds.
    pub(crate) fn git(&self) -> Command {
        let mut git = Command::new("git");
        git.current_dir(&self.src);
        git
    }

    /// Bootstrap embeds a version number into the name of shared libraries it uploads in CI.
    /// Return the version it would have used for the given commit.
    pub(crate) fn artifact_version_part(&self, commit: &str) -> String {
        let (channel, version) = if self.rust_info.is_managed_git_subrepository() {
            let mut channel = self.git();
            channel.arg("show").arg(format!("{}:src/ci/channel", commit));
            let channel = output(&mut channel);
            let mut version = self.git();
            version.arg("show").arg(format!("{}:src/version", commit));
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
                    eprintln!("error: failed to determine artifact channel and/or version");
                    eprintln!(
                        "help: consider using a git checkout or ensure these files are readable"
                    );
                    if let Err(channel) = channel {
                        eprintln!("reading {}/src/ci/channel failed: {:?}", src, channel);
                    }
                    if let Err(version) = version {
                        eprintln!("reading {}/src/version failed: {:?}", src, version);
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

    pub(crate) fn download_rustc_commit(&self) -> Option<&'static str> {
        static DOWNLOAD_RUSTC: OnceCell<Option<String>> = OnceCell::new();
        if self.dry_run() && DOWNLOAD_RUSTC.get().is_none() {
            // avoid trying to actually download the commit
            return None;
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
            println!("{}", msg);
        }
    }

    pub fn sanitizers_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).map(|t| t.sanitizers).flatten().unwrap_or(self.sanitizers)
    }

    pub fn any_sanitizers_enabled(&self) -> bool {
        self.target_config.values().any(|t| t.sanitizers == Some(true)) || self.sanitizers
    }

    pub fn profiler_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).map(|t| t.profiler).flatten().unwrap_or(self.profiler)
    }

    pub fn any_profiler_enabled(&self) -> bool {
        self.target_config.values().any(|t| t.profiler == Some(true)) || self.profiler
    }

    pub fn llvm_enabled(&self) -> bool {
        self.rust_codegen_backends.contains(&INTERNER.intern_str("llvm"))
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

    pub fn default_codegen_backend(&self) -> Option<Interned<String>> {
        self.rust_codegen_backends.get(0).cloned()
    }

    /// Returns the commit to download, or `None` if we shouldn't download CI artifacts.
    fn download_ci_rustc_commit(&self, download_rustc: Option<StringOrBool>) -> Option<String> {
        // If `download-rustc` is not set, default to rebuilding.
        let if_unchanged = match download_rustc {
            None | Some(StringOrBool::Bool(false)) => return None,
            Some(StringOrBool::Bool(true)) => false,
            Some(StringOrBool::String(s)) if s == "if-unchanged" => true,
            Some(StringOrBool::String(other)) => {
                panic!("unrecognized option for download-rustc: {}", other)
            }
        };

        // Handle running from a directory other than the top level
        let top_level = output(self.git().args(&["rev-parse", "--show-toplevel"]));
        let top_level = top_level.trim_end();
        let compiler = format!("{top_level}/compiler/");
        let library = format!("{top_level}/library/");

        // Look for a version to compare to based on the current commit.
        // Only commits merged by bors will have CI artifacts.
        let merge_base = output(
            self.git()
                .arg("rev-list")
                .arg(format!("--author={}", self.stage0_metadata.config.git_merge_commit_email))
                .args(&["-n1", "--first-parent", "HEAD"]),
        );
        let commit = merge_base.trim_end();
        if commit.is_empty() {
            println!("error: could not find commit hash for downloading rustc");
            println!("help: maybe your repository history is too shallow?");
            println!("help: consider disabling `download-rustc`");
            println!("help: or fetch enough history to include one upstream commit");
            crate::detail_exit(1);
        }

        // Warn if there were changes to the compiler or standard library since the ancestor commit.
        let has_changes = !t!(self
            .git()
            .args(&["diff-index", "--quiet", &commit, "--", &compiler, &library])
            .status())
        .success();
        if has_changes {
            if if_unchanged {
                if self.verbose > 0 {
                    println!(
                        "warning: saw changes to compiler/ or library/ since {commit}; \
                            ignoring `download-rustc`"
                    );
                }
                return None;
            }
            println!(
                "warning: `download-rustc` is enabled, but there are changes to \
                    compiler/ or library/"
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
