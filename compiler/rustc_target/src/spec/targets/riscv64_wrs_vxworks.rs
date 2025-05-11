use crate::spec::{StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv64-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: "riscv64".into(),
        options: TargetOptions {
            cpu: "generic-rv64".into(),
            llvm_abiname: "lp64d".into(),
            max_atomic_width: Some(64),
            features: "+m,+a,+f,+d,+c,+zicsr,+zifencei".into(),
            stack_probes: StackProbeType::Inline,
            ..base::vxworks::opts()
        },
    }
}
