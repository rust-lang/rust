use crate::spec::{SanitizerSet, StackProbeType, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-linux-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 Linux (kernel 4.1, glibc 2.17+)".into()),
            tier: Some(1),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+v8a,+outline-atomics".into(),
            mcount: "\u{1}_mcount".into(),
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::Inline,
            supported_sanitizers: SanitizerSet::ADDRESS
                | SanitizerSet::CFI
                | SanitizerSet::KCFI
                | SanitizerSet::LEAK
                | SanitizerSet::MEMORY
                | SanitizerSet::MEMTAG
                | SanitizerSet::THREAD
                | SanitizerSet::HWADDRESS,
            supports_xray: true,
            ..base::linux_gnu::opts()
        },
    }
}
