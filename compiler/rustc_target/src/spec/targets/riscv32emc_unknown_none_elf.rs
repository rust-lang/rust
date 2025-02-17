use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let abi = "ilp32e";
    Target {
        // The below `data_layout` is explicitly specified by the ilp32e ABI in LLVM. See also
        // `options.llvm_abiname`.
        data_layout: "e-m:e-p:32:32-i64:64-n32-S32".into(),
        llvm_target: "riscv32".into(),
        metadata: TargetMetadata {
            description: Some("Bare RISC-V (RV32EMC ISA)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        arch: "riscv32".into(),

        options: TargetOptions {
            abi: abi.into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            cpu: "generic-rv32".into(),
            // The ilp32e ABI specifies the `data_layout`
            llvm_abiname: abi.into(),
            max_atomic_width: Some(32),
            atomic_cas: false,
            features: "+e,+m,+c,+forced-atomics".into(),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            ..Default::default()
        },
    }
}
