use crate::spec::{Cc, LinkerFlavor, Lld, SanitizerSet, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::android_base::opts();
    base.cpu = "x86-64".into();
    // https://developer.android.com/ndk/guides/abis.html#86-64
    base.features = "+mmx,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.stack_probes = StackProbeType::X86;
    base.supports_xray = true;

    Target {
        llvm_target: "x86_64-linux-android".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: TargetOptions { supported_sanitizers: SanitizerSet::ADDRESS, ..base },
    }
}
