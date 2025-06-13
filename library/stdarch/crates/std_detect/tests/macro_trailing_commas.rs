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
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    ),
    feature(stdarch_internal)
)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(
    any(target_arch = "aarch64", target_arch = "arm64ec"),
    feature(stdarch_aarch64_feature_detection)
)]
#![cfg_attr(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    feature(stdarch_powerpc_feature_detection)
)]
#![cfg_attr(target_arch = "s390x", feature(stdarch_s390x_feature_detection))]
#![cfg_attr(
    any(target_arch = "riscv32", target_arch = "riscv64"),
    feature(stdarch_riscv_feature_detection)
)]
#![cfg_attr(
    target_arch = "loongarch64",
    feature(stdarch_loongarch_feature_detection)
)]

#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv32",
    target_arch = "riscv64",
    target_arch = "loongarch64"
))]
#[macro_use]
extern crate std_detect;

#[test]
#[cfg(target_arch = "arm")]
fn arm() {
    let _ = is_arm_feature_detected!("neon");
    let _ = is_arm_feature_detected!("neon",);
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
fn aarch64() {
    let _ = is_aarch64_feature_detected!("fp");
    let _ = is_aarch64_feature_detected!("fp",);
}

#[test]
#[cfg(target_arch = "loongarch64")]
fn loongarch64() {
    let _ = is_loongarch_feature_detected!("lsx");
    let _ = is_loongarch_feature_detected!("lsx",);
}

#[test]
#[cfg(target_arch = "powerpc")]
fn powerpc() {
    let _ = is_powerpc_feature_detected!("altivec");
    let _ = is_powerpc_feature_detected!("altivec",);
}

#[test]
#[cfg(target_arch = "powerpc64")]
fn powerpc64() {
    let _ = is_powerpc64_feature_detected!("altivec");
    let _ = is_powerpc64_feature_detected!("altivec",);
}

#[test]
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn riscv() {
    let _ = is_riscv_feature_detected!("zk");
    let _ = is_riscv_feature_detected!("zk",);
}

#[test]
#[cfg(target_arch = "s390x")]
fn s390x() {
    let _ = is_s390x_feature_detected!("vector");
    let _ = is_s390x_feature_detected!("vector",);
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86() {
    let _ = is_x86_feature_detected!("sse");
    let _ = is_x86_feature_detected!("sse",);
}
