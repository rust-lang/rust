use crate::spec::{Cc, LinkerFlavor, SanitizerSet, StackProbeType, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::solaris::opts();
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::Yes), &["-m64"]);
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.vendor = "pc".into();
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::THREAD;

    Target {
        llvm_target: "x86_64-pc-solaris".into(),
        metadata: TargetMetadata {
            description: Some("64-bit Solaris 11.4".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
