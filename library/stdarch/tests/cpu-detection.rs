#![feature(cfg_target_feature)]
#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "cargo-clippy", allow(option_unwrap_used))]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[macro_use]
extern crate stdsimd;

#[test]
#[cfg(all(target_arch = "arm", target_os = "linux"))]
fn arm_linux() {
    println!("neon: {}", cfg_feature_enabled!("neon"));
    println!("pmull: {}", cfg_feature_enabled!("pmull"));
}

#[test]
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn aarch64_linux() {
    println!("neon: {}", cfg_feature_enabled!("neon"));
    println!("asimd: {}", cfg_feature_enabled!("asimd"));
    println!("pmull: {}", cfg_feature_enabled!("pmull"));
}
