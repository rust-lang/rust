use crate::spec::{Arch, Cc, LinkerFlavor, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::l4re::opts();
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.max_atomic_width = Some(64);
    let extra_link_args = &["-zmax-page-size=0x1000", "-zcommon-page-size=0x1000"];
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::Yes), extra_link_args);
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::No), extra_link_args);

    Target {
        llvm_target: "x86_64-unknown-l4re-gnu".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: Arch::X86_64,
        options: base,
    }
}
