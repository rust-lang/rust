use crate::spec::base::apple::{opts, visionos_llvm_target, Arch};
use crate::spec::{FramePointer, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64;
    let mut base = opts("visionos", arch);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::THREAD;

    Target {
        llvm_target: visionos_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a16".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}
