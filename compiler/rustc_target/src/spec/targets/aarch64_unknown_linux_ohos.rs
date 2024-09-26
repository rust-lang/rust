use crate::spec::{SanitizerSet, StackProbeType, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_ohos::opts();
    base.max_atomic_width = Some(128);

    Target {
        llvm_target: "aarch64-unknown-linux-ohos".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 OpenHarmony".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+reserve-x18".into(),
            mcount: "\u{1}_mcount".into(),
            stack_probes: StackProbeType::Inline,
            supported_sanitizers: SanitizerSet::ADDRESS
                | SanitizerSet::CFI
                | SanitizerSet::LEAK
                | SanitizerSet::MEMORY
                | SanitizerSet::MEMTAG
                | SanitizerSet::THREAD
                | SanitizerSet::HWADDRESS,
            ..base
        },
    }
}
