use rustc_abi::Endian;

use crate::spec::{StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.max_atomic_width = Some(128);

    Target {
        llvm_target: "aarch64_be-unknown-linux-gnu_ilp32".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Linux (big-endian, ILP32 ABI)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            abi: "ilp32".into(),
            features: "+v8a,+outline-atomics".into(),
            stack_probes: StackProbeType::Inline,
            mcount: "\u{1}_mcount".into(),
            endian: Endian::Big,
            ..base
        },
    }
}
