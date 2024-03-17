use crate::spec::base::apple::{opts, pre_link_args, tvos_sim_llvm_target, Arch, TargetAbi};
use crate::spec::{FramePointer, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::Arm64;
    const OS: &'static str = "tvos";
    const ABI: TargetAbi = TargetAbi::Normal;

    Target {
        llvm_target: tvos_sim_llvm_target(ARCH).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a7".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)))
        },
    }
}
