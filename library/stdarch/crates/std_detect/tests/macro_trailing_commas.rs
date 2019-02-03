#![feature(stdsimd)]
#![cfg_attr(stdsimd_strict, deny(warnings))]
#![cfg_attr(
    feature = "cargo-clippy",
    allow(clippy::option_unwrap_used, clippy::use_debug, clippy::print_stdout)
)]

#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "powerpc",
    target_arch = "powerpc64"
))]
#[macro_use]
extern crate std_detect;

#[test]
#[cfg(all(target_arch = "arm", any(target_os = "linux", target_os = "android")))]
fn arm_linux() {
    let _ = is_arm_feature_detected!("neon");
    let _ = is_arm_feature_detected!("neon",);
}

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
fn aarch64_linux() {
    let _ = is_aarch64_feature_detected!("fp");
    let _ = is_aarch64_feature_detected!("fp",);
}

#[test]
#[cfg(all(target_arch = "powerpc", target_os = "linux"))]
fn powerpc_linux() {
    let _ = is_powerpc_feature_detected!("altivec");
    let _ = is_powerpc_feature_detected!("altivec",);
}

#[test]
#[cfg(all(target_arch = "powerpc64", target_os = "linux"))]
fn powerpc64_linux() {
    let _ = is_powerpc64_feature_detected!("altivec");
    let _ = is_powerpc64_feature_detected!("altivec",);
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_all() {
    let _ = is_x86_feature_detected!("sse");
    let _ = is_x86_feature_detected!("sse",);
}
