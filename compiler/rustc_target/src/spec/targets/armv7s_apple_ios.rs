use crate::spec::base::apple::{ios_llvm_target, opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Armv7s;
    Target {
        llvm_target: ios_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".into(),
            max_atomic_width: Some(64),
            ..opts("ios", arch)
        },
    }
}
