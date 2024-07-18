use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

const LINK_SCRIPT: &str = include_str!("./armv7a_vex_v5_linker_script.ld");

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            cpu: "cortex-a9".into(),
            abi: "eabihf".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            link_script: Some(LINK_SCRIPT.into()),
            features: "+v7,+thumb2,+vfp3,+neon".into(),
            relocation_model: RelocModel::Static,
            disable_redzone: true,
            max_atomic_width: Some(64),
            panic_strategy: PanicStrategy::Abort,
            emit_debug_gdb_scripts: false,
            c_enum_min_bits: Some(8),
            os: "vexos".into(),
            vendor: "vex".into(),
            ..Default::default()
        },
    }
}
