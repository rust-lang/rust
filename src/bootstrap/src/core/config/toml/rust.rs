use serde::{Deserialize, Deserializer};

use crate::core::config::toml::common::{DebuginfoLevel, StringOrBool};
use crate::core::config::toml::{Merge, ReplaceOpt};
use crate::{BTreeSet, HashSet, PathBuf, define_config, exit};

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
        tools_debug_assertions: Option<bool> = "debug-assertions-tools",
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

impl<'de> Deserialize<'de> for LldMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LldModeVisitor;

        impl serde::de::Visitor<'_> for LldModeVisitor {
            type Value = LldMode;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
