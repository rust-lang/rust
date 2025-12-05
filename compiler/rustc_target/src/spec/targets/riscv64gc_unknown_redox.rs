use crate::spec::{Arch, CodeModel, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::redox::opts();
    base.code_model = Some(CodeModel::Medium);
    base.cpu = "generic-rv64".into();
    base.features = "+m,+a,+f,+d,+c".into();
    base.llvm_abiname = "lp64d".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "riscv64-unknown-redox".into(),
        metadata: TargetMetadata {
            description: Some("Redox OS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: Arch::RiscV64,
        options: base,
    }
}
