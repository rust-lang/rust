//! This module defines the `Rust` struct, which represents the `[rust]` table
//! in the `bootstrap.toml` configuration file.

use serde::{Deserialize, Deserializer};

use crate::core::config::toml::TomlConfig;
use crate::core::config::{DebuginfoLevel, Merge, ReplaceOpt, StringOrBool};
use crate::{BTreeSet, CodegenBackendKind, HashSet, PathBuf, TargetSelection, define_config, exit};

define_config! {
    /// TOML representation of how the Rust build is configured.
    #[derive(Default)]
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
        break_on_ice: Option<bool> = "break-on-ice",
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

/// Compares the current Rust options against those in the CI rustc builder and detects any incompatible options.
/// It does this by destructuring the `Rust` instance to make sure every `Rust` field is covered and not missing.
pub fn check_incompatible_options_for_ci_rustc(
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
        current_config_toml.build.as_ref().and_then(|b| b.optimized_compiler_builtins.clone());
    let optimized_compiler_builtins =
        ci_config_toml.build.as_ref().and_then(|b| b.optimized_compiler_builtins.clone());
    err!(current_optimized_compiler_builtins, optimized_compiler_builtins, "build");

    // We always build the in-tree compiler on cross targets, so we only care
    // about the host target here.
    let host_str = host.to_string();
    if let Some(current_cfg) = current_config_toml.target.as_ref().and_then(|c| c.get(&host_str))
        && current_cfg.profiler.is_some()
    {
        let ci_target_toml = ci_config_toml.target.as_ref().and_then(|c| c.get(&host_str));
        let ci_cfg = ci_target_toml.ok_or(format!(
            "Target specific config for '{host_str}' is not present for CI-rustc"
        ))?;

        let profiler = &ci_cfg.profiler;
        err!(current_cfg.profiler, profiler, "build");

        let optimized_compiler_builtins = &ci_cfg.optimized_compiler_builtins;
        err!(current_cfg.optimized_compiler_builtins, optimized_compiler_builtins, "build");
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
        default_linker,
        std_features,

        // Rest of the options can simply be ignored.
        incremental: _,
        debug: _,
        codegen_units: _,
        codegen_units_std: _,
        rustc_debug_assertions: _,
        std_debug_assertions: _,
        tools_debug_assertions: _,
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
        break_on_ice: _,
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

    Ok(())
}

pub(crate) const BUILTIN_CODEGEN_BACKENDS: &[&str] = &["llvm", "cranelift", "gcc"];

pub(crate) fn parse_codegen_backends(
    backends: Vec<String>,
    section: &str,
) -> Vec<CodegenBackendKind> {
    const CODEGEN_BACKEND_PREFIX: &str = "rustc_codegen_";

    let mut found_backends = vec![];
    for backend in &backends {
        if let Some(stripped) = backend.strip_prefix(CODEGEN_BACKEND_PREFIX) {
            panic!(
                "Invalid value '{backend}' for '{section}.codegen-backends'. \
                Codegen backends are defined without the '{CODEGEN_BACKEND_PREFIX}' prefix. \
                Please, use '{stripped}' instead."
            )
        }
        if !BUILTIN_CODEGEN_BACKENDS.contains(&backend.as_str()) {
            println!(
                "HELP: '{backend}' for '{section}.codegen-backends' might fail. \
                List of known codegen backends: {BUILTIN_CODEGEN_BACKENDS:?}"
            );
        }
        let backend = match backend.as_str() {
            "llvm" => CodegenBackendKind::Llvm,
            "cranelift" => CodegenBackendKind::Cranelift,
            "gcc" => CodegenBackendKind::Gcc,
            backend => CodegenBackendKind::Custom(backend.to_string()),
        };
        found_backends.push(backend);
    }
    if found_backends.is_empty() {
        eprintln!("ERROR: `{section}.codegen-backends` should not be set to `[]`");
        exit!(1);
    }
    found_backends
}

#[cfg(not(test))]
pub fn default_lld_opt_in_targets() -> Vec<String> {
    vec!["x86_64-unknown-linux-gnu".to_string()]
}

#[cfg(test)]
thread_local! {
    static TEST_LLD_OPT_IN_TARGETS: std::cell::RefCell<Option<Vec<String>>> = std::cell::RefCell::new(None);
}

#[cfg(test)]
pub fn default_lld_opt_in_targets() -> Vec<String> {
    TEST_LLD_OPT_IN_TARGETS.with(|cell| cell.borrow().clone()).unwrap_or_default()
}

#[cfg(test)]
pub fn with_lld_opt_in_targets<R>(targets: Vec<String>, f: impl FnOnce() -> R) -> R {
    TEST_LLD_OPT_IN_TARGETS.with(|cell| {
        let prev = cell.replace(Some(targets));
        let result = f();
        cell.replace(prev);
        result
    })
}
