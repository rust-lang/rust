use crate::spec::{
    FramePointer, SanitizerSet, StackProbeType, Target, TargetMetadata, TargetOptions, base,
};

// See https://developer.android.com/ndk/guides/abis.html#arm64-v8a
// for target ABI requirements.

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64-linux-android".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Android".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            max_atomic_width: Some(128),
            // As documented in https://developer.android.com/ndk/guides/cpu-features.html
            // the neon (ASIMD) and FP must exist on all android aarch64 targets.
            features: "+v8a,+neon,+fp-armv8".into(),
            // the AAPCS64 expects use of non-leaf frame pointers per
            // https://github.com/ARM-software/abi-aa/blob/4492d1570eb70c8fd146623e0db65b2d241f12e7/aapcs64/aapcs64.rst#the-frame-pointer
            // and we tend to encounter interesting bugs in AArch64 unwinding code if we do not
            frame_pointer: FramePointer::NonLeaf,
            stack_probes: StackProbeType::Inline,
            supported_sanitizers: SanitizerSet::CFI
                | SanitizerSet::HWADDRESS
                | SanitizerSet::MEMTAG
                | SanitizerSet::SHADOWCALLSTACK
                | SanitizerSet::ADDRESS,
            supports_xray: true,
            ..base::android::opts()
        },
    }
}
