use crate::spec::base::apple::{base, Arch, TargetAbi};
use crate::spec::{Target, TargetOptions};

pub(crate) fn target() -> Target {
    let (opts, llvm_target, arch) = base("ios", Arch::Armv7s, TargetAbi::Normal);
    Target {
        llvm_target,
        metadata: crate::spec::TargetMetadata {
            description: Some("Armv7-A Apple-A6 Apple iOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32".into(),
        arch,
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".into(),
            max_atomic_width: Some(64),
            ..opts
        },
    }
}
