// Configuration shared with both libm and libm-test

use std::env::{self, VarError};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

/// Read from env, print more debug output via `cargo:warning` if set.
static VERBOSE_BUILD: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
#[allow(dead_code)]
pub struct Config {
    pub manifest_dir: PathBuf,
    pub out_dir: PathBuf,
    pub opt_level: String,
    pub cargo_features: Vec<String>,
    pub target_triple: String,
    pub target_arch: String,
    pub target_env: String,
    pub target_families: Vec<String>,
    pub target_os: String,
    pub target_string: String,
    pub target_vendor: String,
    pub target_features: Vec<String>,
    pub reliable_f128: bool,
    pub reliable_f16: bool,
}

impl Config {
    pub fn from_env() -> Self {
        println!("cargo:cargo::rerun-if-env-changed=LIBM_BUILD_VERBOSE");
        if env_flag("LIBM_BUILD_VERBOSE") {
            VERBOSE_BUILD.store(true, Relaxed);
        }

        let target_triple = env::var("TARGET").unwrap();
        let target_families = env::var("CARGO_CFG_TARGET_FAMILY")
            .map(|feats| feats.split(',').map(ToOwned::to_owned).collect())
            .unwrap_or_default();
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|feats| feats.split(',').map(ToOwned::to_owned).collect())
            .unwrap_or_default();
        let cargo_features = env::vars()
            .filter_map(|(name, _value)| name.strip_prefix("CARGO_FEATURE_").map(ToOwned::to_owned))
            .map(|s| s.to_lowercase().replace("_", "-"))
            .collect();
        if VERBOSE_BUILD.load(Relaxed) {
            for feature in &cargo_features {
                println!("cargo:warning=feature `{feature}` enabled");
            }
        }

        Self {
            target_triple,
            manifest_dir: PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()),
            out_dir: PathBuf::from(env::var("OUT_DIR").unwrap()),
            opt_level: env::var("OPT_LEVEL").unwrap(),
            cargo_features,
            target_arch: env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
            target_env: env::var("CARGO_CFG_TARGET_ENV").unwrap(),
            target_families,
            target_os: env::var("CARGO_CFG_TARGET_OS").unwrap(),
            target_string: env::var("TARGET").unwrap(),
            target_vendor: env::var("CARGO_CFG_TARGET_VENDOR").unwrap(),
            target_features,
            // Note that these are unstable options, so only show up with the nightly compiler or
            // with `RUSTC_BOOTSTRAP=1` (which is required to use the types anyway).
            reliable_f128: env::var_os("CARGO_CFG_TARGET_HAS_RELIABLE_F128").is_some(),
            reliable_f16: env::var_os("CARGO_CFG_TARGET_HAS_RELIABLE_F16").is_some(),
        }
    }
}

/// Libm gets most config options made available.
#[allow(dead_code)]
pub fn emit_libm_config(cfg: &Config) {
    emit_intrinsics_cfg();
    emit_optimization_cfg(cfg);
    emit_cfg_shorthands(cfg);
    emit_cfg_env(cfg);
    emit_f16_f128_cfg(cfg);
}

/// Tests don't need most feature-related config.
#[allow(dead_code)]
pub fn emit_test_config(cfg: &Config) {
    emit_optimization_cfg(cfg);
    emit_cfg_shorthands(cfg);
    emit_cfg_env(cfg);
    emit_f16_f128_cfg(cfg);
}

/// Simplify the feature logic for enabling intrinsics so code only needs to use
/// `cfg(intrinsics_enabled)`.
fn emit_intrinsics_cfg() {
    // Disabled by default; `unstable-intrinsics` enables again; `force-soft-floats` overrides
    // to disable.
    let intrinsics = cfg!(feature = "unstable-intrinsics") && cfg!(feature = "arch");
    set_cfg("intrinsics_enabled", intrinsics);
}

/// Some tests are extremely slow. Emit a config option based on optimization level.
fn emit_optimization_cfg(cfg: &Config) {
    let opt = !matches!(cfg.opt_level.as_str(), "0" | "1");
    set_cfg("optimizations_enabled", opt);
}

/// Provide an alias for common longer config combinations.
fn emit_cfg_shorthands(cfg: &Config) {
    // Shorthand to detect i586 targets
    let x86_no_sse2 = cfg.target_arch == "x86" && !cfg.target_features.iter().any(|f| f == "sse2");
    set_cfg("x86_no_sse2", x86_no_sse2);
}

/// Reemit config that we make use of for test logging.
fn emit_cfg_env(cfg: &Config) {
    println!(
        "cargo:rustc-env=CFG_CARGO_FEATURES={:?}",
        cfg.cargo_features
    );
    println!("cargo:rustc-env=CFG_OPT_LEVEL={}", cfg.opt_level);
    println!(
        "cargo:rustc-env=CFG_TARGET_FEATURES={:?}",
        cfg.target_features
    );
}

/// Configure whether or not `f16` and `f128` support should be enabled.
fn emit_f16_f128_cfg(cfg: &Config) {
    // `unstable-float` enables these features. See the compiler-builtins file for their
    // meaning.
    let unstable_float = cfg!(feature = "unstable-float");
    set_cfg("f16_enabled", unstable_float && cfg.reliable_f16);
    set_cfg("f128_enabled", unstable_float && cfg.reliable_f128);
}

pub fn set_cfg(name: &str, set: bool) {
    println!("cargo:rustc-check-cfg=cfg({name})");
    if !set {
        return;
    }
    if VERBOSE_BUILD.load(Relaxed) {
        println!("cargo:warning=setting config `{name}`");
    }
    println!("cargo:rustc-cfg={name}");
}

/// Return true if the env is set to a value other than `0`.
pub fn env_flag(key: &str) -> bool {
    match env::var(key) {
        Ok(x) if x == "0" => false,
        Err(VarError::NotPresent) => false,
        Err(VarError::NotUnicode(_)) => panic!("non-unicode var for `{key}`"),
        Ok(_) => true,
    }
}
