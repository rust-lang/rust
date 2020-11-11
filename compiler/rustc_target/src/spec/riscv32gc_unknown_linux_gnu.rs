use crate::spec::{CodeModel, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "riscv32-unknown-linux-gnu".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".to_string(),
        arch: "riscv32".to_string(),
        options: TargetOptions {
            unsupported_abis: super::riscv_base::unsupported_abis(),
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv32".to_string(),
            features: "+m,+a,+f,+d,+c".to_string(),
            llvm_abiname: "ilp32d".to_string(),
            max_atomic_width: Some(32),
            ..super::linux_gnu_base::opts()
        },
    }
}
