use crate::spec::base::apple::{opts, pre_link_args, watchos_llvm_target, Arch, TargetAbi};
use crate::spec::{MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const OS: &str = "watchos";
    const ARCH: Arch = Arch::Arm64_32;
    const ABI: TargetAbi = TargetAbi::Normal;

    let base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));

    Target {
        llvm_target: watchos_llvm_target(ARCH).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
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
