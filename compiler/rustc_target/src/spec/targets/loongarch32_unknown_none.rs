use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "loongarch32-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("Freestanding/bare-metal LoongArch32".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        arch: "loongarch32".into(),
        options: TargetOptions {
            cpu: "generic".into(),
            features: "+f,+d".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            llvm_abiname: "ilp32d".into(),
            max_atomic_width: Some(32),
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Abort,
            ..Default::default()
        },
    }
}
