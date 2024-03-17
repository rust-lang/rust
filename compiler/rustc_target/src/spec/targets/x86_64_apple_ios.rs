use crate::spec::base::apple::{ios_sim_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{MaybeLazy, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::X86_64;
    const OS: &'static str = "ios";
    const ABI: TargetAbi = TargetAbi::Simulator;

    // x86_64-apple-ios is a simulator target, even though it isn't declared
    // that way in the target name like the other ones...
    let mut base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::THREAD;

    Target {
        llvm_target: ios_sim_llvm_target(ARCH).into(),
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
        options: TargetOptions { max_atomic_width: Some(128), ..base },
    }
}
