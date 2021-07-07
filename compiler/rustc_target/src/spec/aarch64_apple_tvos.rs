use super::apple_sdk_base::{opts, Arch};
use crate::spec::{FramePointer, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "arm64-apple-tvos".to_string(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a7".to_string(),
            max_atomic_width: Some(128),
            forces_embed_bitcode: true,
            frame_pointer: FramePointer::NonLeaf,
            ..opts("tvos", Arch::Arm64)
        },
    }
}
