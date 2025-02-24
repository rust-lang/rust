use crate::spec::{
    RustcAbi, SanitizerSet, StackProbeType, Target, TargetMetadata, TargetOptions, base,
};

// See https://developer.android.com/ndk/guides/abis.html#x86
// for target ABI requirements.

pub(crate) fn target() -> Target {
    let mut base = base::android::opts();

    base.max_atomic_width = Some(64);

    base.rustc_abi = Some(RustcAbi::X86Sse2);
    // https://developer.android.com/ndk/guides/abis.html#x86
    base.cpu = "pentium4".into();
    base.features = "+mmx,+sse,+sse2,+sse3,+ssse3".into();
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "i686-linux-android".into(),
        metadata: TargetMetadata {
            description: Some("32-bit x86 Android".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: TargetOptions { supported_sanitizers: SanitizerSet::ADDRESS, ..base },
    }
}
