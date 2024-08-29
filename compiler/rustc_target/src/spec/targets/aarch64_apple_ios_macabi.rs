use crate::spec::base::apple::{mac_catalyst_llvm_target, opts, Arch, TargetAbi};
use crate::spec::{FramePointer, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64;
    let mut base = opts("ios", arch, TargetAbi::MacCatalyst);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::LEAK | SanitizerSet::THREAD;

    Target {
        llvm_target: mac_catalyst_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Apple Catalyst on ARM64".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a12".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}
