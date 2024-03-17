use crate::spec::base::apple::{opts, pre_link_args, watchos_sim_llvm_target, Arch, TargetAbi};
use crate::spec::{MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::X86_64;
    const OS: &'static str = "watchos";
    const ABI: TargetAbi = TargetAbi::Simulator;

    Target {
        llvm_target: watchos_sim_llvm_target(ARCH).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            max_atomic_width: Some(128),
            ..opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)))
        },
    }
}
