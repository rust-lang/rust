use crate::spec::base::apple::{opts, tvos_sim_llvm_target, Arch};
use crate::spec::{FramePointer, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64_sim;
    Target {
        llvm_target: tvos_sim_llvm_target(arch).into(),
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
            features: "+neon,+fp-armv8,+apple-a7".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..opts("tvos", arch)
        },
    }
}
