use crate::spec::{Cc, LinkerFlavor, Lld, StackProbeType, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.cpu = "x86-64".into();
    base.abi = "x32".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-mx32"]);
    base.stack_probes = StackProbeType::Inline;
    base.has_thread_local = false;
    // BUG(GabrielMajeri): disabling the PLT on x86_64 Linux with x32 ABI
    // breaks code gen. See LLVM bug 36743
    base.plt_by_default = true;

    Target {
        llvm_target: "x86_64-unknown-linux-gnux32".into(),
        metadata: TargetMetadata {
            description: Some("64-bit Linux (x32 ABI) (kernel 4.15, glibc 2.27)".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-i128:128-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: base,
    }
}
