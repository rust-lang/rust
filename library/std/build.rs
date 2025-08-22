use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH was not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS was not set");
    let target_vendor =
        env::var("CARGO_CFG_TARGET_VENDOR").expect("CARGO_CFG_TARGET_VENDOR was not set");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").expect("CARGO_CFG_TARGET_ENV was not set");

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
        || target_os == "vexos"

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
}
