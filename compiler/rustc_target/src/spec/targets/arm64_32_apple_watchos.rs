use crate::spec::base::apple::{opts, watchos_llvm_target, Arch, TargetAbi};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64_32;
    let base = opts("watchos", arch, TargetAbi::Normal);
    Target {
        llvm_target: watchos_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Arm Apple WatchOS 64-bit with 32-bit pointers".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
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
