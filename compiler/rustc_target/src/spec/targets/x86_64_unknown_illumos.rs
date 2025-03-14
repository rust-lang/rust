use crate::spec::{Cc, LinkerFlavor, SanitizerSet, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::illumos::opts();
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::Yes), &["-m64", "-std=c99"]);
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(64);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::THREAD;

    Target {
        // LLVM does not currently have a separate illumos target,
        // so we still pass Solaris to it
        llvm_target: "x86_64-pc-solaris".into(),
        metadata: TargetMetadata {
            description: Some("illumos".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
