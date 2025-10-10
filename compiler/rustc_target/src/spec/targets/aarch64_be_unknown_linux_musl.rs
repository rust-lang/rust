use rustc_abi::Endian;

use crate::spec::{
    FramePointer, SanitizerSet, StackProbeType, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.max_atomic_width = Some(128);
    base.supports_xray = true;
    base.features = "+v8a,+outline-atomics".into();
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers = SanitizerSet::ADDRESS
        | SanitizerSet::CFI
        | SanitizerSet::LEAK
        | SanitizerSet::MEMORY
        | SanitizerSet::THREAD;

    Target {
        llvm_target: "aarch64_be-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Linux (big-endian) with musl-libc 1.2.5".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            // the AAPCS64 expects use of non-leaf frame pointers per
            // https://github.com/ARM-software/abi-aa/blob/4492d1570eb70c8fd146623e0db65b2d241f12e7/aapcs64/aapcs64.rst#the-frame-pointer
            // and we tend to encounter interesting bugs in AArch64 unwinding code if we do not
            frame_pointer: FramePointer::NonLeaf,
            mcount: "\u{1}_mcount".into(),
            endian: Endian::Big,
            ..base
        },
    }
}
