use crate::spec::{StackProbeType, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv32".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        arch: "riscv32".into(),
        options: TargetOptions {
            cpu: "generic-rv32".into(),
            llvm_abiname: "ilp32d".into(),
            max_atomic_width: Some(32),
            features: "+m,+a,+f,+d,+c".into(),
            stack_probes: StackProbeType::Inline,
            ..base::vxworks::opts()
        },
    }
}
