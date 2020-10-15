use crate::spec::{CodeModel, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "riscv32-unknown-linux-gnu".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        target_env: "gnu".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".to_string(),
        arch: "riscv32".to_string(),
        target_os: "linux".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            unsupported_abis: super::riscv_base::unsupported_abis(),
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv32".to_string(),
            features: "+m,+a,+f,+d,+c".to_string(),
            llvm_abiname: "ilp32d".to_string(),
            max_atomic_width: Some(32),
            ..super::linux_base::opts()
        },
    }
}
