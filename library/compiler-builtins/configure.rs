// Configuration that is shared between `compiler_builtins` and `testcrate`.

use std::env;

#[derive(Debug)]
#[allow(dead_code)]
pub struct Target {
    pub triple: String,
    pub os: String,
    pub arch: String,
    pub vendor: String,
    pub env: String,
    pub pointer_width: u8,
    pub little_endian: bool,
    pub features: Vec<String>,
}

impl Target {
    pub fn from_env() -> Self {
        let little_endian = match env::var("CARGO_CFG_TARGET_ENDIAN").unwrap().as_str() {
            "little" => true,
            "big" => false,
            x => panic!("unknown endian {x}"),
        };

        Self {
            triple: env::var("TARGET").unwrap(),
            os: env::var("CARGO_CFG_TARGET_OS").unwrap(),
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
        }
    }

    #[allow(dead_code)]
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
}

/// Configure whether or not `f16` and `f128` support should be enabled.
pub fn configure_f16_f128(target: &Target) {
    // Set whether or not `f16` and `f128` are supported at a basic level by LLVM. This only means
    // that the backend will not crash when using these types. This does not mean that the
    // backend does the right thing, or that the platform doesn't have ABI bugs.
    //
    // We do this here rather than in `rust-lang/rust` because configuring via cargo features is
    // not straightforward.
    //
    // Original source of this list:
    // <https://github.com/rust-lang/compiler-builtins/pull/652#issuecomment-2266151350>
    let (f16_ok, f128_ok) = match target.arch.as_str() {
        // `f16` and `f128` both crash <https://github.com/llvm/llvm-project/issues/94434>
        "arm64ec" => (false, false),
        // `f16` crashes <https://github.com/llvm/llvm-project/issues/50374>
        "s390x" => (false, true),
        // FIXME(llvm): `f16` test failures fixed by <https://github.com/llvm/llvm-project/pull/107791>
        "loongarch64" => (false, true),
        // `f128` crashes <https://github.com/llvm/llvm-project/issues/96432>
        "mips64" | "mips64r6" => (true, false),
        // `f128` crashes <https://github.com/llvm/llvm-project/issues/101545>
        "powerpc64" if &target.os == "aix" => (true, false),
        // `f128` crashes <https://github.com/llvm/llvm-project/issues/41838>
        "sparc" => (true, false),
        // `f16` miscompiles <https://github.com/llvm/llvm-project/issues/96438>
        "wasm32" | "wasm64" => (false, true),
        // Most everything else works as of LLVM 19
        _ => (true, true),
    };

    // If the feature is set, disable these types.
    let disable_both = env::var_os("CARGO_FEATURE_NO_F16_F128").is_some();

    println!("cargo::rustc-check-cfg=cfg(f16_enabled)");
    println!("cargo::rustc-check-cfg=cfg(f128_enabled)");

    if f16_ok && !disable_both {
        println!("cargo::rustc-cfg=f16_enabled");
    }

    if f128_ok && !disable_both {
        println!("cargo::rustc-cfg=f128_enabled");
    }
}
