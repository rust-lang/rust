use crate::spec::{
    base, Cc, LinkerFlavor, MaybeLazy, SanitizerSet, StackProbeType, Target, TargetOptions,
};

pub fn target() -> Target {
    let mut base = base::solaris::opts();
    base.pre_link_args =
        MaybeLazy::lazy(|| TargetOptions::link_args(LinkerFlavor::Unix(Cc::Yes), &["-m64"]));
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.vendor = "pc".into();
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::THREAD;

    Target {
        llvm_target: "x86_64-pc-solaris".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
