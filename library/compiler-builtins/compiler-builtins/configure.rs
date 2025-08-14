// Configuration that is shared between `compiler_builtins` and `builtins_test`.

use std::{env, str};

#[derive(Debug)]
#[allow(dead_code)]
pub struct Target {
    pub triple: String,
    pub triple_split: Vec<String>,
    pub opt_level: String,
    pub cargo_features: Vec<String>,
    pub os: String,
    pub arch: String,
    pub vendor: String,
    pub env: String,
    pub pointer_width: u8,
    pub little_endian: bool,
    pub features: Vec<String>,
    pub reliable_f128: bool,
    pub reliable_f16: bool,
}

impl Target {
    pub fn from_env() -> Self {
        let triple = env::var("TARGET").unwrap();
        let triple_split = triple.split('-').map(ToOwned::to_owned).collect();
        let little_endian = match env::var("CARGO_CFG_TARGET_ENDIAN").unwrap().as_str() {
            "little" => true,
            "big" => false,
            x => panic!("unknown endian {x}"),
        };
        let cargo_features = env::vars()
            .filter_map(|(name, _value)| name.strip_prefix("CARGO_FEATURE_").map(ToOwned::to_owned))
            .map(|s| s.to_lowercase().replace("_", "-"))
            .collect();

        Self {
            triple,
            triple_split,
            os: env::var("CARGO_CFG_TARGET_OS").unwrap(),
            opt_level: env::var("OPT_LEVEL").unwrap(),
            cargo_features,
            arch: env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
            vendor: env::var("CARGO_CFG_TARGET_VENDOR").unwrap(),
            env: env::var("CARGO_CFG_TARGET_ENV").unwrap(),
            pointer_width: env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
                .unwrap()
                .parse()
                .unwrap(),
            little_endian,
            features: env::var("CARGO_CFG_TARGET_FEATURE")
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
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
}

pub fn configure_aliases(target: &Target) {
    // To compile builtins-test-intrinsics for thumb targets, where there is no libc
    println!("cargo::rustc-check-cfg=cfg(thumb)");
    if target.triple_split[0].starts_with("thumb") {
        println!("cargo:rustc-cfg=thumb")
    }

    // compiler-rt `cfg`s away some intrinsics for thumbv6m and thumbv8m.base because
    // these targets do not have full Thumb-2 support but only original Thumb-1.
    // We have to cfg our code accordingly.
    println!("cargo::rustc-check-cfg=cfg(thumb_1)");
    if target.triple_split[0] == "thumbv6m" || target.triple_split[0] == "thumbv8m.base" {
        println!("cargo:rustc-cfg=thumb_1")
    }

    // Config shorthands
    println!("cargo:rustc-check-cfg=cfg(x86_no_sse)");
    if target.arch == "x86" && !target.features.iter().any(|f| f == "sse") {
        // Shorthand to detect i586 targets
        println!("cargo:rustc-cfg=x86_no_sse");
    }

    /* Not all backends support `f16` and `f128` to the same level on all architectures, so we
     * need to disable things if the compiler may crash. See configuration at:
     * * https://github.com/rust-lang/rust/blob/c65dccabacdfd6c8a7f7439eba13422fdd89b91e/compiler/rustc_codegen_llvm/src/llvm_util.rs#L367-L432
     * * https://github.com/rust-lang/rustc_codegen_gcc/blob/4b5c44b14166083eef8d71f15f5ea1f53fc976a0/src/lib.rs#L496-L507
     * * https://github.com/rust-lang/rustc_codegen_cranelift/blob/c713ffab3c6e28ab4b4dd4e392330f786ea657ad/src/lib.rs#L196-L226
     */

    // If the feature is set, disable both of these types.
    let no_f16_f128 = target.cargo_features.iter().any(|s| s == "no-f16-f128");

    println!("cargo::rustc-check-cfg=cfg(f16_enabled)");
    if target.reliable_f16 && !no_f16_f128 {
        println!("cargo::rustc-cfg=f16_enabled");
    }

    println!("cargo::rustc-check-cfg=cfg(f128_enabled)");
    if target.reliable_f128 && !no_f16_f128 {
        println!("cargo::rustc-cfg=f128_enabled");
    }
}
