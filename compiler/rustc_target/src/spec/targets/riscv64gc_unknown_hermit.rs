use crate::spec::{CodeModel, RelocModel, Target, TargetMetadata, TargetOptions, TlsModel, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv64-unknown-hermit".into(),
        metadata: TargetMetadata {
            description: Some("RISC-V Hermit".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        arch: "riscv64".into(),
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        options: TargetOptions {
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c,+zicsr,+zifencei".into(),
            relocation_model: RelocModel::Pic,
            code_model: Some(CodeModel::Medium),
            tls_model: TlsModel::LocalExec,
            max_atomic_width: Some(64),
            llvm_abiname: "lp64d".into(),
            ..base::hermit::opts()
        },
    }
}
