// This portion of configuration is shared between `std` and `coretests`.

use std::env;

#[allow(dead_code)] // Not all importers of this file use all fields
pub struct Config {
    pub target_arch: String,
    pub target_os: String,
    pub target_vendor: String,
    pub target_env: String,
    pub target_abi: String,
    pub target_pointer_width: u32,
    pub target_features: Vec<String>,
    pub is_miri: bool,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            target_arch: env::var("CARGO_CFG_TARGET_ARCH")
                .expect("CARGO_CFG_TARGET_ARCH was not set"),
            target_os: env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS was not set"),
            target_vendor: env::var("CARGO_CFG_TARGET_VENDOR")
                .expect("CARGO_CFG_TARGET_VENDOR was not set"),
            target_env: env::var("CARGO_CFG_TARGET_ENV").expect("CARGO_CFG_TARGET_ENV was not set"),
            target_abi: env::var("CARGO_CFG_TARGET_ABI").expect("CARGO_CFG_TARGET_ABI was not set"),
            target_pointer_width: env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
                .expect("CARGO_CFG_TARGET_POINTER_WIDTH was not set")
                .parse()
                .unwrap(),
            target_features: env::var("CARGO_CFG_TARGET_FEATURE")
                .unwrap_or_default()
                .split(",")
                .map(ToOwned::to_owned)
                .collect(),
            is_miri: env::var_os("CARGO_CFG_MIRI").is_some(),
        }
    }
}

pub fn configure_f16_f128(cfg: &Config) {
    // Emit these on platforms that have no known ABI bugs, LLVM selection bugs, lowering bugs,
    // missing symbols, or other problems, to determine when tests get run.
    // If more broken platforms are found, please update the tracking issue at
    // <https://github.com/rust-lang/rust/issues/116909>
    //
    // Some of these match arms are redundant; the goal is to separate reasons that the type is
    // unreliable, even when multiple reasons might fail the same platform.
    println!("cargo:rustc-check-cfg=cfg(reliable_f16)");
    println!("cargo:rustc-check-cfg=cfg(reliable_f128)");

    // This is a step beyond only having the types and basic functions available. Math functions
    // aren't consistently available or correct.
    println!("cargo:rustc-check-cfg=cfg(reliable_f16_math)");
    println!("cargo:rustc-check-cfg=cfg(reliable_f128_math)");

    let target_arch = cfg.target_arch.as_str();
    let target_os = cfg.target_os.as_str();

    let has_reliable_f16 = match (target_arch, target_os) {
        // We can always enable these in Miri as that is not affected by codegen bugs.
        _ if cfg.is_miri => true,
        // Selection failure <https://github.com/llvm/llvm-project/issues/50374>
        ("s390x", _) => false,
        // Unsupported <https://github.com/llvm/llvm-project/issues/94434>
        ("arm64ec", _) => false,
        // MinGW ABI bugs <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115054>
        ("x86_64", "windows") if cfg.target_env == "gnu" && cfg.target_abi != "llvm" => false,
        // Infinite recursion <https://github.com/llvm/llvm-project/issues/97981>
        ("csky", _) => false,
        ("hexagon", _) => false,
        ("loongarch64", _) => false,
        ("mips" | "mips64" | "mips32r6" | "mips64r6", _) => false,
        ("powerpc" | "powerpc64", _) => false,
        ("sparc" | "sparc64", _) => false,
        ("wasm32" | "wasm64", _) => false,
        // `f16` support only requires that symbols converting to and from `f32` are available. We
        // provide these in `compiler-builtins`, so `f16` should be available on all platforms that
        // do not have other ABI issues or LLVM crashes.
        _ => true,
    };

    let has_reliable_f128 = match (target_arch, target_os) {
        // We can always enable these in Miri as that is not affected by codegen bugs.
        _ if cfg.is_miri => true,
        // Unsupported <https://github.com/llvm/llvm-project/issues/94434>
        ("arm64ec", _) => false,
        // Selection bug <https://github.com/llvm/llvm-project/issues/96432>
        ("mips64" | "mips64r6", _) => false,
        // Selection bug <https://github.com/llvm/llvm-project/issues/95471>
        ("nvptx64", _) => false,
        // ABI bugs <https://github.com/rust-lang/rust/issues/125109> et al. (full
        // list at <https://github.com/rust-lang/rust/issues/116909>)
        ("powerpc" | "powerpc64", _) => false,
        // ABI unsupported  <https://github.com/llvm/llvm-project/issues/41838>
        ("sparc", _) => false,
        // Stack alignment bug <https://github.com/llvm/llvm-project/issues/77401>. NB: tests may
        // not fail if our compiler-builtins is linked.
        ("x86", _) => false,
        // MinGW ABI bugs <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115054>
        ("x86_64", "windows") if cfg.target_env == "gnu" && cfg.target_abi != "llvm" => false,
        // There are no known problems on other platforms, so the only requirement is that symbols
        // are available. `compiler-builtins` provides all symbols required for core `f128`
        // support, so this should work for everything else.
        _ => true,
    };

    // Configure platforms that have reliable basics but may have unreliable math.

    // LLVM is currently adding missing routines, <https://github.com/llvm/llvm-project/issues/93566>
    let has_reliable_f16_math = has_reliable_f16
        && match (target_arch, target_os) {
            // FIXME: Disabled on Miri as the intrinsics are not implemented yet.
            _ if cfg.is_miri => false,
            // x86 has a crash for `powi`: <https://github.com/llvm/llvm-project/issues/105747>
            ("x86" | "x86_64", _) => false,
            // Assume that working `f16` means working `f16` math for most platforms, since
            // operations just go through `f32`.
            _ => true,
        };

    let has_reliable_f128_math = has_reliable_f128
        && match (target_arch, target_os) {
            // FIXME: Disabled on Miri as the intrinsics are not implemented yet.
            _ if cfg.is_miri => false,
            // LLVM lowers `fp128` math to `long double` symbols even on platforms where
            // `long double` is not IEEE binary128. See
            // <https://github.com/llvm/llvm-project/issues/44744>.
            //
            // This rules out anything that doesn't have `long double` = `binary128`; <= 32 bits
            // (ld is `f64`), anything other than Linux (Windows and MacOS use `f64`), and `x86`
            // (ld is 80-bit extended precision).
            ("x86_64", _) => false,
            (_, "linux") if cfg.target_pointer_width == 64 => true,
            _ => false,
        };

    if has_reliable_f16 {
        println!("cargo:rustc-cfg=reliable_f16");
    }
    if has_reliable_f128 {
        println!("cargo:rustc-cfg=reliable_f128");
    }
    if has_reliable_f16_math {
        println!("cargo:rustc-cfg=reliable_f16_math");
    }
    if has_reliable_f128_math {
        println!("cargo:rustc-cfg=reliable_f128_math");
    }
}
