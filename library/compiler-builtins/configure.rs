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
    // that the backend will not crash when using these types and generates code that can be called
    // without crashing (no infinite recursion). This does not mean that the platform doesn't have
    // ABI or other bugs.
    //
    // We do this here rather than in `rust-lang/rust` because configuring via cargo features is
    // not straightforward.
    //
    // Original source of this list:
    // <https://github.com/rust-lang/compiler-builtins/pull/652#issuecomment-2266151350>
    let f16_enabled = match target.arch.as_str() {
        // Unsupported <https://github.com/llvm/llvm-project/issues/94434>
        "arm64ec" => false,
        // Selection failure <https://github.com/llvm/llvm-project/issues/50374>
        "s390x" => false,
        // Infinite recursion <https://github.com/llvm/llvm-project/issues/97981>
        // FIXME(llvm): loongarch fixed by <https://github.com/llvm/llvm-project/pull/107791>
        "csky" => false,
        "hexagon" => false,
        "loongarch64" => false,
        "mips" | "mips64" | "mips32r6" | "mips64r6" => false,
        "powerpc" | "powerpc64" => false,
        "sparc" | "sparc64" => false,
        "wasm32" | "wasm64" => false,
        // Most everything else works as of LLVM 19
        _ => true,
    };

    let f128_enabled = match target.arch.as_str() {
        // Unsupported (libcall is not supported) <https://github.com/llvm/llvm-project/issues/121122>
        "amdgpu" => false,
        // Unsupported <https://github.com/llvm/llvm-project/issues/94434>
        "arm64ec" => false,
        // Selection failure <https://github.com/llvm/llvm-project/issues/96432>
        "mips64" | "mips64r6" => false,
        // Selection failure <https://github.com/llvm/llvm-project/issues/95471>
        "nvptx64" => false,
        // Selection failure <https://github.com/llvm/llvm-project/issues/101545>
        "powerpc64" if &target.os == "aix" => false,
        // Selection failure <https://github.com/llvm/llvm-project/issues/41838>
        "sparc" => false,
        // Most everything else works as of LLVM 19
        _ => true,
    };

    // If the feature is set, disable these types.
    let disable_both = env::var_os("CARGO_FEATURE_NO_F16_F128").is_some();

    println!("cargo::rustc-check-cfg=cfg(f16_enabled)");
    println!("cargo::rustc-check-cfg=cfg(f128_enabled)");

    if f16_enabled && !disable_both {
        println!("cargo::rustc-cfg=f16_enabled");
    }

    if f128_enabled && !disable_both {
        println!("cargo::rustc-cfg=f128_enabled");
    }
}
