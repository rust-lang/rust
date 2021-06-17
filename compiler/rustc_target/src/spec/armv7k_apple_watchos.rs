use super::apple_sdk_base::{opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let base = opts("watchos", Arch::Armv7k);
    Target {
        llvm_target: "armv7k-apple-watchos".to_string(),
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-i64:64-a:0:32-n32-S128".to_string(),
        arch: "arm".to_string(),
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".to_string(),
            max_atomic_width: Some(64),
            unsupported_abis: super::arm_base::unsupported_abis(),
            ..base
        },
    }
}
