use crate::spec::{CodeModel, Target, TargetOptions};
use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel};

pub fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n64-S128".to_string(),
        llvm_target: "riscv64".to_string(),
        pointer_width: 64,
        arch: "riscv64".to_string(),

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
            linker: Some("rust-lld".to_string()),
            cpu: "generic-rv64".to_string(),
            max_atomic_width: Some(64),
            atomic_cas: true,
            features: "+m,+a,+c".to_string(),
            executables: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            code_model: Some(CodeModel::Medium),
            emit_debug_gdb_scripts: false,
            unsupported_abis: super::riscv_base::unsupported_abis(),
            eh_frame_header: false,
            ..Default::default()
        },
    }
}
