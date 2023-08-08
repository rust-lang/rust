use crate::spec::{Cc, LinkerFlavor, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::solaris_base::opts();
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::Yes), &["-m64"]);
    base.cpu = "x86-64".into();
    base.plt_by_default = false;
    base.vendor = "sun".into();
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::X86;

    Target {
        llvm_target: "x86_64-pc-solaris".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: base,
    }
}
