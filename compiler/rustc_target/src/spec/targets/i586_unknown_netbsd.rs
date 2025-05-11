use crate::spec::{StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::netbsd::opts();
    base.cpu = "pentium".into();
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "i586-unknown-netbsdelf".into(),
        metadata: TargetMetadata {
            description: Some("32-bit x86, resricted to Pentium".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: TargetOptions { mcount: "__mcount".into(), ..base },
    }
}
