use crate::spec::{Arch, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::helenos::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a".into();
    base.linker = Some("aarch64-helenos-gcc".into());

    Target {
        llvm_target: "aarch64-unknown-helenos".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 HelenOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: Arch::AArch64,
        options: base,
    }
}
