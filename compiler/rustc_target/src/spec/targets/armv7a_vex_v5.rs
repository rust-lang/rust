use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

const LINK_SCRIPT: &str = include_str!("./armv7a_vex_v5_linker_script.ld");

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Armv7-A Cortex-A9 VEX V5 Brain, VEXos".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            os: "vexos".into(),
            vendor: "vex".into(),
            exe_suffix: ".elf".into(),
            cpu: "cortex-a9".into(),
            abi: "eabihf".into(),
            features: "+v7,+neon,+vfp3,+thumb2,+thumb-mode".into(),
            linker: Some("rust-lld".into()),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            link_script: Some(LINK_SCRIPT.into()),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            c_enum_min_bits: Some(8),
            max_atomic_width: Some(64),
            disable_redzone: true,
            emit_debug_gdb_scripts: false,
            has_thumb_interworking: true,
            ..Default::default()
        },
    }
}
