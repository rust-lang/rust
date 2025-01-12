#![allow(internal_features)]
#![cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ),
    feature(stdarch_internal)
)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_feature_detection))]
#![cfg_attr(target_arch = "powerpc", feature(stdarch_powerpc_feature_detection))]
#![cfg_attr(target_arch = "powerpc64", feature(stdarch_powerpc_feature_detection))]
#![cfg_attr(target_arch = "s390x", feature(stdarch_s390x_feature_detection))]
#![allow(clippy::unwrap_used, clippy::use_debug, clippy::print_stdout)]

#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x",
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
#[cfg(all(target_arch = "s390x", target_os = "linux"))]
fn s390x_linux() {
    let _ = is_s390x_feature_detected!("vector");
    let _ = is_s390x_feature_detected!("vector",);
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_all() {
    let _ = is_x86_feature_detected!("sse");
    let _ = is_x86_feature_detected!("sse",);
}
