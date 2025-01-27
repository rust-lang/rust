use crate::spec::{Cc, LinkerFlavor, Lld, RelocModel, StackProbeType, Target, TargetOptions, cvs};

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
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m32"]);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "i686-unknown-nuttx".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("NuttX/x86".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
