use crate::spec::{Arch, CodeModel, LlvmAbi, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv64-unknown-managarm-mlibc".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("managarm/riscv64".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: Arch::RiscV64,
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c".into(),
            llvm_abiname: LlvmAbi::Lp64d,
            max_atomic_width: Some(64),
            ..base::managarm_mlibc::opts()
        },
    }
}
