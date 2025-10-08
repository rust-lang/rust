use crate::spec::{Cc, LinkerFlavor, Lld, RelocModel, Target, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        arch: "avr".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        data_layout: "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8:16-a:8".into(),
        llvm_target: "avr-unknown-unknown".into(),
        pointer_width: 16,
        options: TargetOptions {
            c_int_width: 16,
            exe_suffix: ".elf".into(),
            linker: Some("avr-gcc".into()),
            eh_frame_header: false,
            pre_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &[]),
            late_link_args: TargetOptions::link_args(
                LinkerFlavor::Gnu(Cc::Yes, Lld::No),
                &["-lgcc"],
            ),
            max_atomic_width: Some(16),
            atomic_cas: false,
            relocation_model: RelocModel::Static,
            need_explicit_cpu: true,
            ..TargetOptions::default()
        },
    }
}
