use super::apple_sdk_base::{opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let base = opts("tvos", Arch::Arm64);
    Target {
        llvm_target: "arm64-apple-tvos".to_string(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a7".to_string(),
            eliminate_frame_pointer: false,
            max_atomic_width: Some(128),
            unsupported_abis: super::arm_base::unsupported_abis(),
            forces_embed_bitcode: true,
            ..base
        },
    }
}
