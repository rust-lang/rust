use crate::spec::base::apple::{ios_sim_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::I386;
    const OS: &'static str = "ios";
    // i386-apple-ios is a simulator target, even though it isn't declared
    // that way in the target name like the other ones...
    const ABI: TargetAbi = TargetAbi::Simulator;

    Target {
        // Clang automatically chooses a more specific target based on
        // IPHONEOS_DEPLOYMENT_TARGET.
        // This is required for the target to pick the right
        // MACH-O commands, so we do too.
        llvm_target: ios_sim_llvm_target(ARCH).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            max_atomic_width: Some(64),
            ..opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)))
        },
    }
}
