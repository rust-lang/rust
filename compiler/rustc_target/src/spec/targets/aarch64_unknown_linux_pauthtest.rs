use crate::spec::{
    Arch, Env, FramePointer, LinkSelfContainedDefault, StackProbeType, Target, TargetMetadata,
    TargetOptions, base,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-linux-pauthtest".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Linux with pauth enabled musl".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: Arch::AArch64,

        options: TargetOptions {
            env: Env::Pauthtest,
            features: "+v8.3a,+outline-atomics,+pauth".into(),
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::Inline,
            crt_static_default: false,
            crt_static_allows_dylibs: false,
            // the AAPCS64 expects use of non-leaf frame pointers per
            // https://github.com/ARM-software/abi-aa/blob/4492d1570eb70c8fd146623e0db65b2d241f12e7/aapcs64/aapcs64.rst#the-frame-pointer
            // and we tend to encounter interesting bugs in AArch64 unwinding code if we do not
            frame_pointer: FramePointer::NonLeaf,
            link_self_contained: LinkSelfContainedDefault::False,
            mcount: "\u{1}_mcount".into(),
            ..base::linux_musl::opts()
         },
    }
}
