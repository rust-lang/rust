use crate::spec::{Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a,+neon,+fp-armv8".into();

    Target {
        llvm_target: "aarch64-pc-windows-msvc".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 Windows MSVC".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
