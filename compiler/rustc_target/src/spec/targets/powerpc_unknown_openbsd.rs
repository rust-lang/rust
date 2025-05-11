use rustc_abi::Endian;

use crate::spec::{StackProbeType, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::openbsd::opts();
    base.endian = Endian::Big;
    base.max_atomic_width = Some(32);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc-unknown-openbsd".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-Fn32-i64:64-n32".into(),
        arch: "powerpc".into(),
        options: base,
    }
}
