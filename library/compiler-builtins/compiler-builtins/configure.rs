// Configuration that is shared between `compiler_builtins` and `builtins_test`.

use std::env::{self, VarError};
use std::str;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

/// Read from env, print more debug output via `cargo:warning` if set.
static VERBOSE_BUILD: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
#[allow(dead_code)]
pub struct Config {
    pub cargo_features: Vec<String>,
    pub opt_level: String,
    pub reliable_f128: bool,
    pub reliable_f16: bool,
    pub target_arch: String,
    pub target_env: String,
    pub target_features: Vec<String>,
    pub target_os: String,
    pub target_pointer_width: u8,
    pub target_triple: String,
    pub target_triple_split: Vec<String>,
    pub target_vendor: String,
}

impl Config {
    pub fn from_env() -> Self {
        println!("cargo:cargo::rerun-if-env-changed=LIBM_BUILD_VERBOSE");
        if env_flag("LIBM_BUILD_VERBOSE") {
            VERBOSE_BUILD.store(true, Relaxed);
        }

        let target_triple = env::var("TARGET").unwrap();
        let target_triple_split = target_triple.split('-').map(ToOwned::to_owned).collect();
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
            target_triple_split,
            target_os: env::var("CARGO_CFG_TARGET_OS").unwrap(),
            opt_level: env::var("OPT_LEVEL").unwrap(),
            cargo_features,
            target_arch: env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
            target_vendor: env::var("CARGO_CFG_TARGET_VENDOR").unwrap(),
            target_env: env::var("CARGO_CFG_TARGET_ENV").unwrap(),
            target_pointer_width: env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
                .unwrap()
                .parse()
                .unwrap(),
            target_features: env::var("CARGO_CFG_TARGET_FEATURE")
                .unwrap_or_default()
                .split(",")
                .map(ToOwned::to_owned)
                .collect(),
            // Note that these are unstable options, so only show up with the nightly compiler or
            // with `RUSTC_BOOTSTRAP=1` (which is required to use the types anyway).
            reliable_f128: env::var_os("CARGO_CFG_TARGET_HAS_RELIABLE_F128").is_some(),
            reliable_f16: env::var_os("CARGO_CFG_TARGET_HAS_RELIABLE_F16").is_some(),
        }
    }

    #[allow(dead_code)]
    pub fn has_target_feature(&self, feature: &str) -> bool {
        self.target_features.iter().any(|f| f == feature)
    }
}

pub fn configure_aliases(cfg: &Config) {
    let triple_split = &cfg.target_triple_split;

    // To compile builtins-test-intrinsics for thumb targets, where there is no libc
    let thumb = triple_split[0].starts_with("thumb");
    set_cfg("thumb", thumb);

    // compiler-rt `cfg`s away some intrinsics for thumbv6m and thumbv8m.base because
    // these targets do not have full Thumb-2 support but only original Thumb-1.
    // We have to cfg our code accordingly.
    let thumb_1 = triple_split[0] == "thumbv6m" || triple_split[0] == "thumbv8m.base";
    set_cfg("thumb_1", thumb_1);

    // Shorthand to detect i586 targets
    let x86_no_sse2 = cfg.target_arch == "x86" && !cfg.target_features.iter().any(|f| f == "sse2");
    set_cfg("x86_no_sse2", x86_no_sse2);

    /* Not all backends support `f16` and `f128` to the same level on all architectures, so we
     * need to disable things if the compiler may crash. See configuration at:
     * * https://github.com/rust-lang/rust/blob/c65dccabacdfd6c8a7f7439eba13422fdd89b91e/compiler/rustc_codegen_llvm/src/llvm_util.rs#L367-L432
     * * https://github.com/rust-lang/rustc_codegen_gcc/blob/4b5c44b14166083eef8d71f15f5ea1f53fc976a0/src/lib.rs#L496-L507
     * * https://github.com/rust-lang/rustc_codegen_cranelift/blob/c713ffab3c6e28ab4b4dd4e392330f786ea657ad/src/lib.rs#L196-L226
     */

    set_cfg("f16_enabled", cfg.reliable_f16);
    set_cfg("f128_enabled", cfg.reliable_f128);
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
