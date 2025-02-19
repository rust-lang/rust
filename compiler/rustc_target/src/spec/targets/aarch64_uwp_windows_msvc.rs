use crate::spec::{Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_uwp_msvc::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a".into();

    Target {
        llvm_target: "aarch64-pc-windows-msvc".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
                .into(),
        arch: "aarch64".into(),
        options: base,
    }
}
