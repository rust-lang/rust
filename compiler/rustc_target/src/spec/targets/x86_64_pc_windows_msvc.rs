use crate::spec::{SanitizerSet, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.cpu = "x86-64".into();
    base.features = "+cx16,+sse3,+sahf".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(128);
    base.supported_sanitizers = SanitizerSet::ADDRESS;

    Target {
        llvm_target: "x86_64-pc-windows-msvc".into(),
        metadata: TargetMetadata {
            description: Some("64-bit MSVC (Windows 10+)".into()),
            tier: Some(1),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
