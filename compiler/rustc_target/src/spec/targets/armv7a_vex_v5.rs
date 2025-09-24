use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions,
};

const LINKER_SCRIPT: &str = include_str!("./armv7a_vex_v5_linker_script.ld");

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        vendor: "vex".into(),
        env: "v5".into(),
        os: "vexos".into(),
        cpu: "cortex-a9".into(),
        abi: "eabihf".into(),
        is_like_vexos: true,
        llvm_floatabi: Some(FloatAbi::Hard),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        features: "+v7,+neon,+vfp3d16,+thumb2".into(),
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Abort,
        emit_debug_gdb_scripts: false,
        c_enum_min_bits: Some(8),
        default_uwtable: true,
        has_thumb_interworking: true,
        link_script: Some(LINKER_SCRIPT.into()),
        ..Default::default()
    };
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("ARMv7-A Cortex-A9 VEX V5 Brain".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: opts,
    }
}
