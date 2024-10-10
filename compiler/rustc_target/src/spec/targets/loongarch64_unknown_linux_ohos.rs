use crate::spec::{SanitizerSet, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "loongarch64-unknown-linux-ohos".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("LoongArch64 OpenHarmony".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: "loongarch64".into(),
        options: TargetOptions {
            cpu: "generic".into(),
            features: "+f,+d".into(),
            llvm_abiname: "lp64d".into(),
            max_atomic_width: Some(64),
            supported_sanitizers: SanitizerSet::ADDRESS
                | SanitizerSet::CFI
                | SanitizerSet::LEAK
                | SanitizerSet::MEMORY
                | SanitizerSet::THREAD,
            ..base::linux_ohos::opts()
        },
    }
}
