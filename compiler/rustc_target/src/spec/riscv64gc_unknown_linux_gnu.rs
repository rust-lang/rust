use crate::spec::{CodeModel, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "riscv64-unknown-linux-gnu".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n64-S128".to_string(),
        arch: "riscv64".to_string(),
        options: TargetOptions {
            unsupported_abis: super::riscv_base::unsupported_abis(),
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv64".to_string(),
            features: "+m,+a,+f,+d,+c".to_string(),
            llvm_abiname: "lp64d".to_string(),
            max_atomic_width: Some(64),
            ..super::linux_base::opts()
        },
    }
}
