use crate::spec::{FloatAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "arm-unknown-linux-gnueabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv6 Linux, hardfloat (kernel 3.2, glibc 2.17)".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            features: "+strict-align,+v6,+vfp2,-d32".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}__gnu_mcount_nc".into(),
            llvm_mcount_intrinsic: Some("llvm.arm.gnu.eabi.mcount".into()),
            // The default on linux is to have `default_uwtable=true`, but on
            // this target we get an "`__aeabi_unwind_cpp_pr0` not defined"
            // linker error, so set it to `true` here.
            // FIXME(#146996): Remove this override once #146996 has been fixed.
            default_uwtable: false,
            ..base::linux_gnu::opts()
        },
    }
}
