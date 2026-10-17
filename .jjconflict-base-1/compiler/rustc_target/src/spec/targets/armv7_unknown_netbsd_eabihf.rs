use crate::spec::{Arch, CfgAbi, FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "arm-unknown-netbsdelf-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A NetBSD w/hard-float".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: Arch::Arm,
        options: TargetOptions {
            cfg_abi: CfgAbi::EabiHf,
            llvm_floatabi: Some(FloatAbi::Hard),
            features: "+v7,+db,+dsp,+aclass,+perfmon,+vfp3d16,+thumb2".into(),
            max_atomic_width: Some(64),
            mcount: "__mcount".into(),
            ..base::netbsd::opts()
        },
    }
}
