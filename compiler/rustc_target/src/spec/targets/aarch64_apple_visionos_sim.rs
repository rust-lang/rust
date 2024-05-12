use crate::spec::base::apple::{opts, pre_link_args, visionos_sim_llvm_target, Arch, TargetAbi};
use crate::spec::{FramePointer, MaybeLazy, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    const OS: &str = "visionos";
    const ABI: TargetAbi = TargetAbi::Simulator;
    const ARCH: Arch = Arch::Arm64;

    let mut base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::THREAD;

    Target {
        llvm_target: MaybeLazy::lazy(|| visionos_sim_llvm_target(ARCH)),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 Apple visionOS simulator".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a16".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}
