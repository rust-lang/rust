use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7-unknown-linux-gnueabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A for VxWorks".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            // Info about features at https://wiki.debian.org/ArmHardFloatPort
            features: "+v7,+vfp3,-d32,+thumb2,-neon".into(),
            max_atomic_width: Some(64),
            ..base::vxworks::opts()
        },
    }
}
