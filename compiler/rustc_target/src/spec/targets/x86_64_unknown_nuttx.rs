// Generic x86-64 target for NuttX RTOS
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.

use crate::spec::{RelocModel, StackProbeType, Target, TargetOptions, cvs};

pub(crate) fn target() -> Target {
    let mut base = TargetOptions {
        os: "nuttx".into(),
        dynamic_linking: false,
        families: cvs!["unix"],
        no_default_libraries: true,
        has_rpath: false,
        position_independent_executables: false,
        relocation_model: RelocModel::Static,
        relro_level: crate::spec::RelroLevel::Full,
        has_thread_local: true,
        use_ctors_section: true,
        ..Default::default()
    };
    base.cpu = "x86-64".into();
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "x86_64-unknown-nuttx".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("NuttX/x86_64".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
