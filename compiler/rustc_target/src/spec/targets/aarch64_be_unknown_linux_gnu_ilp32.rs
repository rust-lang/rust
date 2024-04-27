use crate::abi::Endian;
use crate::spec::{base, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.max_atomic_width = Some(128);

    Target {
        llvm_target: "aarch64_be-unknown-linux-gnu_ilp32".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
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
