use crate::spec::base::apple::{opts, pre_link_args, Arch};
use crate::spec::{MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::Arm64_32;
    const OS: &'static str = "watchos";

    let base = opts(OS, ARCH, MaybeLazy::lazy(|| pre_link_args(OS, ARCH)));
    Target {
        llvm_target: "arm64_32-apple-watchos".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-i64:64-i128:128-n32:64-S128".into(),
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
