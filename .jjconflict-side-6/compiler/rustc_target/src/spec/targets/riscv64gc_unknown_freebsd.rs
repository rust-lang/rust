use crate::spec::{Arch, CodeModel, LlvmAbi, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv64-unknown-freebsd".into(),
        metadata: TargetMetadata {
            description: Some("RISC-V FreeBSD".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: Arch::RiscV64,
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c,+zicsr,+zifencei".into(),
            llvm_abiname: LlvmAbi::Lp64d,
            max_atomic_width: Some(64),
            ..base::freebsd::opts()
        },
    }
}
