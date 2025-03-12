use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH was not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS was not set");
    let target_vendor =
        env::var("CARGO_CFG_TARGET_VENDOR").expect("CARGO_CFG_TARGET_VENDOR was not set");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").expect("CARGO_CFG_TARGET_ENV was not set");
    let target_abi = env::var("CARGO_CFG_TARGET_ABI").expect("CARGO_CFG_TARGET_ABI was not set");
    let target_pointer_width: u32 = env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
        .expect("CARGO_CFG_TARGET_POINTER_WIDTH was not set")
        .parse()
        .unwrap();
    let target_features: Vec<_> = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(",")
        .map(ToOwned::to_owned)
        .collect();
    let is_miri = env::var_os("CARGO_CFG_MIRI").is_some();

    println!("cargo:rustc-check-cfg=cfg(netbsd10)");
    if target_os == "netbsd" && env::var("RUSTC_STD_NETBSD10").is_ok() {
        println!("cargo:rustc-cfg=netbsd10");
    }

    println!("cargo:rustc-check-cfg=cfg(restricted_std)");
    if target_os == "linux"
        || target_os == "android"
        || target_os == "netbsd"
        || target_os == "dragonfly"
        || target_os == "openbsd"
        || target_os == "freebsd"
        || target_os == "solaris"
        || target_os == "illumos"
        || target_os == "macos"
        || target_os == "ios"
        || target_os == "tvos"
        || target_os == "watchos"
        || target_os == "visionos"
        || target_os == "windows"
        || target_os == "fuchsia"
        || (target_vendor == "fortanix" && target_env == "sgx")
        || target_os == "hermit"
        || target_os == "trusty"
        || target_os == "l4re"
        || target_os == "redox"
        || target_os == "haiku"
        || target_os == "vxworks"
        || target_arch == "wasm32"
        || target_arch == "wasm64"
        || target_os == "espidf"
        || target_os.starts_with("solid")
        || (target_vendor == "nintendo" && target_env == "newlib")
        || target_os == "vita"
        || target_os == "aix"
        || target_os == "nto"
        || target_os == "xous"
        || target_os == "hurd"
        || target_os == "uefi"
        || target_os == "teeos"
        || target_os == "zkvm"
        || target_os == "rtems"
        || target_os == "nuttx"
        || target_os == "cygwin"

        // See src/bootstrap/src/core/build_steps/synthetic_targets.rs
        || env::var("RUSTC_BOOTSTRAP_SYNTHETIC_TARGET").is_ok()
    {
        // These platforms don't have any special requirements.
    } else {
        // This is for Cargo's build-std support, to mark std as unstable for
        // typically no_std platforms.
        // This covers:
        // - os=none ("bare metal" targets)
        // - mipsel-sony-psp
        // - nvptx64-nvidia-cuda
        // - arch=avr
        // - JSON targets
        // - Any new targets that have not been explicitly added above.
        println!("cargo:rustc-cfg=restricted_std");
    }

    println!("cargo:rustc-check-cfg=cfg(backtrace_in_libstd)");
    println!("cargo:rustc-cfg=backtrace_in_libstd");

    println!("cargo:rustc-env=STD_ENV_ARCH={}", env::var("CARGO_CFG_TARGET_ARCH").unwrap());

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

    let has_reliable_f16 = match (target_arch.as_str(), target_os.as_str()) {
        // We can always enable these in Miri as that is not affected by codegen bugs.
        _ if is_miri => true,
        // Selection failure <https://github.com/llvm/llvm-project/issues/50374>
        ("s390x", _) => false,
        // Unsupported <https://github.com/llvm/llvm-project/issues/94434>
        ("arm64ec", _) => false,
        // LLVM crash <https://github.com/llvm/llvm-project/issues/129394>
        ("aarch64", _) if !target_features.iter().any(|f| f == "neon") => false,
        // MinGW ABI bugs <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115054>
        ("x86_64", "windows") if target_env == "gnu" && target_abi != "llvm" => false,
        // Infinite recursion <https://github.com/llvm/llvm-project/issues/97981>
        ("csky", _) => false,
        ("hexagon", _) => false,
        ("powerpc" | "powerpc64", _) => false,
        ("sparc" | "sparc64", _) => false,
        ("wasm32" | "wasm64", _) => false,
        // `f16` support only requires that symbols converting to and from `f32` are available. We
        // provide these in `compiler-builtins`, so `f16` should be available on all platforms that
        // do not have other ABI issues or LLVM crashes.
        _ => true,
    };

    let has_reliable_f128 = match (target_arch.as_str(), target_os.as_str()) {
        // We can always enable these in Miri as that is not affected by codegen bugs.
        _ if is_miri => true,
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
        ("x86_64", "windows") if target_env == "gnu" && target_abi != "llvm" => false,
        // There are no known problems on other platforms, so the only requirement is that symbols
        // are available. `compiler-builtins` provides all symbols required for core `f128`
        // support, so this should work for everything else.
        _ => true,
    };

    // Configure platforms that have reliable basics but may have unreliable math.

    // LLVM is currently adding missing routines, <https://github.com/llvm/llvm-project/issues/93566>
    let has_reliable_f16_math = has_reliable_f16
        && match (target_arch.as_str(), target_os.as_str()) {
            // FIXME: Disabled on Miri as the intrinsics are not implemented yet.
            _ if is_miri => false,
            // x86 has a crash for `powi`: <https://github.com/llvm/llvm-project/issues/105747>
            ("x86" | "x86_64", _) => false,
            // Assume that working `f16` means working `f16` math for most platforms, since
            // operations just go through `f32`.
            _ => true,
        };

    let has_reliable_f128_math = has_reliable_f128
        && match (target_arch.as_str(), target_os.as_str()) {
            // FIXME: Disabled on Miri as the intrinsics are not implemented yet.
            _ if is_miri => false,
            // LLVM lowers `fp128` math to `long double` symbols even on platforms where
            // `long double` is not IEEE binary128. See
            // <https://github.com/llvm/llvm-project/issues/44744>.
            //
            // This rules out anything that doesn't have `long double` = `binary128`; <= 32 bits
            // (ld is `f64`), anything other than Linux (Windows and MacOS use `f64`), and `x86`
            // (ld is 80-bit extended precision).
            ("x86_64", _) => false,
            (_, "linux") if target_pointer_width == 64 => true,
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
