use crate::spec::{CodeModel, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv64-unknown-netbsd".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("RISC-V NetBSD".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: "riscv64".into(),
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c".into(),
            llvm_abiname: "lp64d".into(),
            max_atomic_width: Some(64),
            mcount: "__mcount".into(),
            ..base::netbsd::opts()
        },
    }
}
