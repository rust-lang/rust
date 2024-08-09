use crate::spec::{cvs, Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "armv7a-unknown-none-eabihf".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            os: "rtems".into(),
            families: cvs!["unix"],
            abi: "eabihf".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            linker: Some("arm-rtems6-gcc".into()),
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Abort,
            features: "+thumb2,+neon,+vfp3".into(),
            max_atomic_width: Some(64),
            emit_debug_gdb_scripts: false,
            // GCC defaults to 8 for arm-none here.
            c_enum_min_bits: Some(8),
            eh_frame_header: false,
            no_default_libraries: false,
            env: "newlib".into(),
            pre_link_args: TargetOptions::link_args(
                LinkerFlavor::Gnu(Cc::Yes, Lld::No),
                &["-qrtems", "-march=armv7-a", "-mthumb", "-mfpu=neon", "-mfloat-abi=hard"],
            ),
            ..Default::default()
        },
    }
}
