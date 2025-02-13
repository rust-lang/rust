use crate::spec::{SanitizerSet, StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.max_atomic_width = Some(128);
    base.supports_xray = true;
    base.features = "+v8a".into();
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers = SanitizerSet::ADDRESS
        | SanitizerSet::CFI
        | SanitizerSet::LEAK
        | SanitizerSet::MEMORY
        | SanitizerSet::THREAD;

    // FIXME(compiler-team#422): musl targets should be dynamically linked by default.
    base.crt_static_default = true;

    Target {
        llvm_target: "aarch64-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Linux with musl 1.2.3".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions { mcount: "\u{1}_mcount".into(), ..base },
    }
}
