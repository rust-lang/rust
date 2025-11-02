use crate::spec::base::apple::{Arch, TargetEnv, base};
use crate::spec::{Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let (opts, llvm_target, arch) = base("watchos", Arch::Arm64, TargetEnv::Normal);
    Target {
        llvm_target,
        metadata: TargetMetadata {
            description: Some("ARM64 Apple watchOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
            .into(),
        arch,
        options: TargetOptions {
            features: "+v8a,+neon,+fp-armv8,+apple-a7".into(),
            max_atomic_width: Some(128),
            dynamic_linking: false,
            position_independent_executables: true,
            ..opts
        },
    }
}
