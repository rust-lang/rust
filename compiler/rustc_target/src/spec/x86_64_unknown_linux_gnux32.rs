use crate::spec::{Cc, LinkerFlavor, Lld, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::linux_gnu_base::opts();
    base.cpu = "x86-64".into();
    base.abi = "x32".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-mx32"]);
    base.stack_probes = StackProbeType::X86;
    base.has_thread_local = false;
    // BUG(GabrielMajeri): disabling the PLT on x86_64 Linux with x32 ABI
    // breaks code gen. See LLVM bug 36743
    base.plt_by_default = true;

    Target {
        llvm_target: "x86_64-unknown-linux-gnux32".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: base,
    }
}
