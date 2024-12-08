use crate::spec::{Cc, LinkerFlavor, Lld, SanitizerSet, StackProbeType, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.supported_sanitizers = SanitizerSet::ADDRESS;
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m32"]);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "i686-unknown-linux-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("32-bit Linux (kernel 3.2, glibc 2.17+)".into()),
            tier: Some(1),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
