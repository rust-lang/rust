use crate::spec::{FloatAbi, PanicStrategy, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv7a-pc-windows-msvc".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            llvm_floatabi: Some(FloatAbi::Hard),
            features: "+vfp3,+neon".into(),
            max_atomic_width: Some(64),
            // FIXME(jordanrh): use PanicStrategy::Unwind when SEH is
            // implemented for windows/arm in LLVM
            panic_strategy: PanicStrategy::Abort,
            ..base::windows_uwp_msvc::opts()
        },
    }
}
