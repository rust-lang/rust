use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        llvm_target: "riscv32".into(),
        metadata: TargetMetadata {
            description: Some("RISC-V Xous (RV32IMAC ISA)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        arch: "riscv32".into(),

        options: TargetOptions {
            os: "xous".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            cpu: "generic-rv32".into(),
            max_atomic_width: Some(32),
            features: "+m,+a,+c".into(),
            llvm_abiname: "ilp32".into(),
            panic_strategy: PanicStrategy::Unwind,
            relocation_model: RelocModel::Static,
            ..Default::default()
        },
    }
}
