//! This module defines the `Rust` struct, which represents the `[rust]` table
//! in the `bootstrap.toml` configuration file.

use std::str::FromStr;

use serde::{Deserialize, Deserializer};

use crate::core::build_steps::compile::CODEGEN_BACKEND_PREFIX;
use crate::core::config::toml::TomlConfig;
use crate::core::config::{
    DebuginfoLevel, Merge, ReplaceOpt, RustcLto, StringOrBool, set, threads_from_config,
};
use crate::flags::Warnings;
use crate::{BTreeSet, Config, HashSet, PathBuf, TargetSelection, define_config, exit};

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
        current_config_toml.build.as_ref().and_then(|b| b.optimized_compiler_builtins);
    let optimized_compiler_builtins =
        ci_config_toml.build.as_ref().and_then(|b| b.optimized_compiler_builtins);
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
        description,
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

    Ok(())
}

impl Config {
    pub fn apply_rust_config(
        &mut self,
        toml_rust: Option<Rust>,
        warnings: Warnings,
        description: &mut Option<String>,
    ) {
        let mut debug = None;
        let mut rustc_debug_assertions = None;
        let mut std_debug_assertions = None;
        let mut tools_debug_assertions = None;
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

        if let Some(rust) = toml_rust {
            let Rust {
                optimize: optimize_toml,
                debug: debug_toml,
                codegen_units,
                codegen_units_std,
                rustc_debug_assertions: rustc_debug_assertions_toml,
                std_debug_assertions: std_debug_assertions_toml,
                tools_debug_assertions: tools_debug_assertions_toml,
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

            if debug_assertions_requested
                && let Some(ref opt) = download_rustc
                && opt.is_string_or_true()
            {
                eprintln!(
                    "WARN: currently no CI rustc builds have rustc debug assertions \
                            enabled. Please either set `rust.debug-assertions` to `false` if you \
                            want to use download CI rustc or set `rust.download-rustc` to `false`."
                );
            }

            self.download_rustc_commit = self.download_ci_rustc_commit(
                download_rustc,
                debug_assertions_requested,
                self.llvm_assertions,
            );

            debug = debug_toml;
            rustc_debug_assertions = rustc_debug_assertions_toml;
            std_debug_assertions = std_debug_assertions_toml;
            tools_debug_assertions = tools_debug_assertions_toml;
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
            self.rust_new_symbol_mangling = new_symbol_mangling;
            set(&mut self.rust_optimize_tests, optimize_tests);
            set(&mut self.codegen_tests, codegen_tests);
            set(&mut self.rust_rpath, rpath);
            set(&mut self.rust_strip, strip);
            set(&mut self.rust_frame_pointers, frame_pointers);
            self.rust_stack_protector = stack_protector;
            set(&mut self.jemalloc, jemalloc);
            set(&mut self.test_compare_mode, test_compare_mode);
            set(&mut self.backtrace, backtrace);
            if rust_description.is_some() {
                eprintln!(
                    "Warning: rust.description is deprecated. Use build.description instead."
                );
            }
            if description.is_none() {
                *description = rust_description;
            }
            set(&mut self.rust_dist_src, dist_src);
            set(&mut self.verbose_tests, verbose_tests);
            // in the case "false" is set explicitly, do not overwrite the command line args
            if let Some(true) = incremental {
                self.incremental = true;
            }
            set(&mut self.lld_mode, lld_mode);
            set(&mut self.llvm_bitcode_linker_enabled, llvm_bitcode_linker);

            self.rust_randomize_layout = randomize_layout.unwrap_or_default();
            self.llvm_tools_enabled = llvm_tools.unwrap_or(true);

            self.llvm_enzyme = self.channel == "dev" || self.channel == "nightly";
            self.rustc_default_linker = default_linker;
            self.musl_root = musl_root.map(PathBuf::from);
            self.save_toolstates = save_toolstates.map(PathBuf::from);
            set(
                &mut self.deny_warnings,
                match warnings {
                    Warnings::Deny => Some(true),
                    Warnings::Warn => Some(false),
                    Warnings::Default => deny_warnings,
                },
            );
            set(&mut self.backtrace_on_ice, backtrace_on_ice);
            set(&mut self.rust_verify_llvm_ir, verify_llvm_ir);
            self.rust_thin_lto_import_instr_limit = thin_lto_import_instr_limit;
            set(&mut self.rust_remap_debuginfo, remap_debuginfo);
            set(&mut self.control_flow_guard, control_flow_guard);
            set(&mut self.ehcont_guard, ehcont_guard);
            self.llvm_libunwind_default =
                llvm_libunwind.map(|v| v.parse().expect("failed to parse rust.llvm-libunwind"));

            if let Some(ref backends) = codegen_backends {
                let available_backends = ["llvm", "cranelift", "gcc"];

                self.rust_codegen_backends = backends.iter().map(|s| {
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

            self.rust_codegen_units = codegen_units.map(threads_from_config);
            self.rust_codegen_units_std = codegen_units_std.map(threads_from_config);

            if self.rust_profile_use.is_none() {
                self.rust_profile_use = profile_use;
            }

            if self.rust_profile_generate.is_none() {
                self.rust_profile_generate = profile_generate;
            }

            self.rust_lto =
                lto.as_deref().map(|value| RustcLto::from_str(value).unwrap()).unwrap_or_default();
            self.rust_validate_mir_opts = validate_mir_opts;
        }

        self.rust_optimize = optimize.unwrap_or(RustOptimize::Bool(true));

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
        if self.host_target.triple == "x86_64-unknown-linux-gnu"
            && self.hosts == [self.host_target]
            && (self.channel == "dev" || self.channel == "nightly")
        {
            let no_llvm_config = self
                .target_config
                .get(&self.host_target)
                .is_some_and(|target_config| target_config.llvm_config.is_none());
            let enable_lld = self.llvm_from_ci || no_llvm_config;
            // Prefer the config setting in case an explicit opt-out is needed.
            self.lld_enabled = lld_enabled.unwrap_or(enable_lld);
        } else {
            set(&mut self.lld_enabled, lld_enabled);
        }

        let default_std_features = BTreeSet::from([String::from("panic-unwind")]);
        self.rust_std_features = std_features.unwrap_or(default_std_features);

        let default = debug == Some(true);
        self.rustc_debug_assertions = rustc_debug_assertions.unwrap_or(default);
        self.std_debug_assertions = std_debug_assertions.unwrap_or(self.rustc_debug_assertions);
        self.tools_debug_assertions = tools_debug_assertions.unwrap_or(self.rustc_debug_assertions);
        self.rust_overflow_checks = overflow_checks.unwrap_or(default);
        self.rust_overflow_checks_std = overflow_checks_std.unwrap_or(self.rust_overflow_checks);

        self.rust_debug_logging = debug_logging.unwrap_or(self.rustc_debug_assertions);

        let with_defaults = |debuginfo_level_specific: Option<_>| {
            debuginfo_level_specific.or(debuginfo_level).unwrap_or(if debug == Some(true) {
                DebuginfoLevel::Limited
            } else {
                DebuginfoLevel::None
            })
        };
        self.rust_debuginfo_level_rustc = with_defaults(debuginfo_level_rustc);
        self.rust_debuginfo_level_std = with_defaults(debuginfo_level_std);
        self.rust_debuginfo_level_tools = with_defaults(debuginfo_level_tools);
        self.rust_debuginfo_level_tests = debuginfo_level_tests.unwrap_or(DebuginfoLevel::None);
    }
}
