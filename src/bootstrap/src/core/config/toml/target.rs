//! This module defines the structures and logic for handling target-specific configuration
//! within the `bootstrap.toml` file. This allows you to customize build settings, tools,
//! and flags for individual compilation targets.
//!
//! It includes:
//!
//! * [`TomlTarget`]: This struct directly mirrors the `[target.<triple>]` sections in your
//!   `bootstrap.toml`. It's used for deserializing raw TOML data for a specific target.
//! * [`Target`]: This struct represents the processed and validated configuration for a
//!   build target, which is is stored in the main [`Config`] structure.
//! * [`Config::apply_target_config`]: This method processes the `TomlTarget` data and
//!   applies it to the global [`Config`], ensuring proper path resolution, validation,
//!   and integration with other build settings.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer};

use crate::core::build_steps::compile::CODEGEN_BACKEND_PREFIX;
use crate::core::config::{LlvmLibunwind, Merge, ReplaceOpt, SplitDebuginfo, StringOrBool};
use crate::{Config, HashSet, PathBuf, TargetSelection, define_config, exit};

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

impl Config {
    pub fn apply_target_config(&mut self, toml_target: Option<HashMap<String, TomlTarget>>) {
        if let Some(t) = toml_target {
            for (triple, cfg) in t {
                let mut target = Target::from_triple(&triple);

                if let Some(ref s) = cfg.llvm_config {
                    if self.download_rustc_commit.is_some() && triple == *self.build.triple {
                        panic!(
                            "setting llvm_config for the host is incompatible with download-rustc"
                        );
                    }
                    target.llvm_config = Some(self.src.join(s));
                }
                if let Some(patches) = cfg.llvm_has_rust_patches {
                    assert!(
                        self.submodules == Some(false) || cfg.llvm_config.is_some(),
                        "use of `llvm-has-rust-patches` is restricted to cases where either submodules are disabled or llvm-config been provided"
                    );
                    target.llvm_has_rust_patches = Some(patches);
                }
                if let Some(ref s) = cfg.llvm_filecheck {
                    target.llvm_filecheck = Some(self.src.join(s));
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

                self.target_config.insert(TargetSelection::from_user(&triple), target);
            }
        }
    }
}
