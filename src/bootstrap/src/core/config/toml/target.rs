//! This module defines the structures and logic for handling target-specific configuration
//! within the `bootstrap.toml` file. This allows you to customize build settings, tools,
//! and flags for individual compilation targets.
//!
//! It includes:
//!
//! * [`TomlTarget`]: This struct directly mirrors the `[target.<triple>]` sections in your
//!   `bootstrap.toml`. It's used for deserializing raw TOML data for a specific target.
//! * [`Target`]: This struct represents the processed and validated configuration for a
//!   build target, which is is stored in the main `Config` structure.

use std::collections::HashMap;

use serde::de::Error;
use serde::{Deserialize, Deserializer};

use crate::core::config::{
    CompilerBuiltins, LlvmLibunwind, Merge, ReplaceOpt, SplitDebuginfo, StringOrBool,
};
use crate::{CodegenBackendKind, HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how each build target is configured.
    struct TomlTarget {
        cc: Option<String> = "cc",
        cxx: Option<String> = "cxx",
        ar: Option<String> = "ar",
        ranlib: Option<String> = "ranlib",
        default_linker: Option<PathBuf> = "default-linker",
        default_linker_linux_override: Option<DefaultLinuxLinkerOverride> = "default-linker-linux-override",
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
        optimized_compiler_builtins: Option<CompilerBuiltins> = "optimized-compiler-builtins",
        jemalloc: Option<bool> = "jemalloc",
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
    pub default_linker_linux_override: DefaultLinuxLinkerOverride,
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
    pub codegen_backends: Option<Vec<CodegenBackendKind>>,
    pub optimized_compiler_builtins: Option<CompilerBuiltins>,
    pub jemalloc: Option<bool>,
}

impl Target {
    pub fn from_triple(triple: &str) -> Self {
        let mut target: Self = Default::default();
        if !build_helper::targets::target_supports_std(triple) {
            target.no_std = true;
        }
        if triple.contains("emscripten") {
            target.runner = Some("node".into());
        }
        target
    }
}

/// Overrides the default linker used on a Linux target.
/// On Linux, the linker is usually invoked through `cc`, therefore this exists as a separate
/// configuration from simply setting `default-linker`, which corresponds to `-Clinker`.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum DefaultLinuxLinkerOverride {
    /// Do not apply any override and use the default linker for the given target.
    #[default]
    Off,
    /// Use the self-contained `rust-lld` linker, invoked through `cc`.
    /// Corresponds to `-Clinker-features=+lld -Clink-self-contained=+linker`.
    SelfContainedLldCc,
}

impl<'de> Deserialize<'de> for DefaultLinuxLinkerOverride {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        match name.as_str() {
            "off" => Ok(Self::Off),
            "self-contained-lld-cc" => Ok(Self::SelfContainedLldCc),
            other => Err(D::Error::unknown_variant(other, &["off", "self-contained-lld-cc"])),
        }
    }
}

/// Set of linker overrides for selected Linux targets.
#[cfg(not(test))]
pub fn default_linux_linker_overrides() -> HashMap<String, DefaultLinuxLinkerOverride> {
    [("x86_64-unknown-linux-gnu".to_string(), DefaultLinuxLinkerOverride::SelfContainedLldCc)]
        .into()
}

#[cfg(test)]
thread_local! {
    static TEST_LINUX_LINKER_OVERRIDES: std::cell::RefCell<Option<HashMap<String, DefaultLinuxLinkerOverride>>> = std::cell::RefCell::new(None);
}

#[cfg(test)]
pub fn default_linux_linker_overrides() -> HashMap<String, DefaultLinuxLinkerOverride> {
    TEST_LINUX_LINKER_OVERRIDES.with(|cell| cell.borrow().clone()).unwrap_or_default()
}

#[cfg(test)]
pub fn with_default_linux_linker_overrides<R>(
    targets: HashMap<String, DefaultLinuxLinkerOverride>,
    f: impl FnOnce() -> R,
) -> R {
    TEST_LINUX_LINKER_OVERRIDES.with(|cell| {
        let prev = cell.replace(Some(targets));
        let result = f();
        cell.replace(prev);
        result
    })
}
