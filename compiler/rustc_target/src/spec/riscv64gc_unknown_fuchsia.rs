use crate::spec::{CodeModel, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "riscv64gc-unknown-fuchsia".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: "riscv64".into(),
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c".into(),
            llvm_abiname: "lp64d".into(),
            max_atomic_width: Some(64),
            supported_sanitizers: SanitizerSet::SHADOWCALLSTACK,
            ..super::fuchsia_base::opts()
        },
    }
}
