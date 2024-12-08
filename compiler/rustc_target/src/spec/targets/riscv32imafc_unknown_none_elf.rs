use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        llvm_target: "riscv32".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Bare RISC-V (RV32IMAFC ISA)".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        arch: "riscv32".into(),

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            cpu: "generic-rv32".into(),
            max_atomic_width: Some(32),
            llvm_abiname: "ilp32f".into(),
            features: "+m,+a,+c,+f".into(),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            ..Default::default()
        },
    }
}
