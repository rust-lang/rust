use crate::spec::base::apple::{Arch, TargetAbi, base};
use crate::spec::{FramePointer, SanitizerSet, Target, TargetOptions};

pub(crate) fn target() -> Target {
    let (opts, llvm_target, arch) = base("ios", Arch::Arm64, TargetAbi::Simulator);
    Target {
        llvm_target,
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 Apple iOS Simulator".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
            .into(),
        arch,
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a7".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            supported_sanitizers: SanitizerSet::ADDRESS | SanitizerSet::THREAD,
            ..opts
        },
    }
}
