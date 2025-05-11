use crate::spec::{RelocModel, StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let base = base::solid::opts("asp3");
    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 SOLID with TOPPERS/ASP3".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            linker: Some("aarch64-kmc-elf-gcc".into()),
            features: "+v8a,+neon,+fp-armv8".into(),
            relocation_model: RelocModel::Static,
            disable_redzone: true,
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::Inline,
            ..base
        },
    }
}
