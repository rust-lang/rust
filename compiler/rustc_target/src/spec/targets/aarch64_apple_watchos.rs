use crate::spec::base::apple::{opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let base = opts("watchos", Arch::Arm64);
    Target {
        llvm_target: "aarch64-apple-watchos".into(),
        description: None,
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+v8a,+neon,+fp-armv8,+apple-a7".into(),
            max_atomic_width: Some(128),
            dynamic_linking: false,
            position_independent_executables: true,
            ..base
        },
    }
}
