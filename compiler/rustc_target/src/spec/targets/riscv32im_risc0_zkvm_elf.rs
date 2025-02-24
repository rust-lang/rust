use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        llvm_target: "riscv32".into(),
        metadata: TargetMetadata {
            description: Some("RISC Zero's zero-knowledge Virtual Machine (RV32IM ISA)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        arch: "riscv32".into(),

        options: TargetOptions {
            os: "zkvm".into(),
            vendor: "risc0".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            cpu: "generic-rv32".into(),

            // Some crates (*cough* crossbeam) assume you have 64 bit
            // atomics if the target name is not in a hardcoded list.
            // Since zkvm is singlethreaded and all operations are
            // atomic, I guess we can just say we support 64-bit
            // atomics.
            max_atomic_width: Some(64),
            atomic_cas: true,

            features: "+m".into(),
            llvm_abiname: "ilp32".into(),
            executables: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            singlethread: true,
            ..Default::default()
        },
    }
}
