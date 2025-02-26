use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "arm-unknown-linux-gnueabi".into(),
        metadata: TargetMetadata {
            description: Some("Armv6 Linux (kernel 3.2, glibc 2.17)".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            features: "+strict-align,+v6".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}__gnu_mcount_nc".into(),
            llvm_mcount_intrinsic: Some("llvm.arm.gnu.eabi.mcount".into()),
            ..base::linux_gnu::opts()
        },
    }
}
