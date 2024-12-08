use crate::spec::{Cc, LinkerFlavor, Lld, SanitizerSet, Target, TargetOptions, base};

// This target if is for the baseline of the Android v7a ABI
// in thumb mode. It's named armv7-* instead of thumbv7-*
// for historical reasons. See the thumbv7neon variant for
// enabling NEON.

// See https://developer.android.com/ndk/guides/abis.html#v7a
// for target ABI requirements.

pub(crate) fn target() -> Target {
    let mut base = base::android::opts();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-march=armv7-a"]);
    Target {
        llvm_target: "armv7-none-linux-android".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Armv7-A Android".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            features: "+v7,+thumb-mode,+thumb2,+vfp3,-d32,-neon".into(),
            supported_sanitizers: SanitizerSet::ADDRESS,
            max_atomic_width: Some(64),
            ..base
        },
    }
}
