use crate::spec::{CodeModel, SanitizerSet, StackProbeType, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::fuchsia::opts();
    base.code_model = Some(CodeModel::Medium);
    base.cpu = "generic-rv64".into();
    base.features = "+m,+a,+f,+d,+c,+zicsr,+zifencei".into();
    base.llvm_abiname = "lp64d".into();
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers = SanitizerSet::SHADOWCALLSTACK;
    base.supports_xray = true;

    Target {
        llvm_target: "riscv64-unknown-fuchsia".into(),
        metadata: TargetMetadata {
            description: Some("RISC-V Fuchsia".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: "riscv64".into(),
        options: base,
    }
}
