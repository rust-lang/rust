use crate::spec::{
    Arch, Cc, LinkerFlavor, Lld, Os, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        llvm_target: "riscv32".into(),
        metadata: TargetMetadata {
            description: Some("OpenVM zero-knowledge Virtual Machine (RV32IM ISA)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        arch: Arch::RiscV32,

        options: TargetOptions {
            os: Os::Openvm,
            vendor: "unknown".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            cpu: "generic-rv32".into(),

            // We set atomic_width to 64 for compatibility with crates such as crossbeam,
            // but this should never be triggered since compilation should always lower
            // atomics and be single-threaded.
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
