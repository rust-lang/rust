use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions, cvs,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7-unknown-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv7 RTEMS (Requires RTEMS toolchain and kernel".into()),
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
            llvm_floatabi: Some(FloatAbi::Hard),
            linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            linker: None,
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Unwind,
            features: "+thumb2,+neon,+vfp3".into(),
            max_atomic_width: Some(64),
            emit_debug_gdb_scripts: false,
            // GCC defaults to 8 for arm-none here.
            c_enum_min_bits: Some(8),
            eh_frame_header: false,
            no_default_libraries: false,
            env: "newlib".into(),
            ..Default::default()
        },
    }
}
