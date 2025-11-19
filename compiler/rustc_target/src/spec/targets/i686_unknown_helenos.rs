use crate::spec::{Arch, Cc, LinkerFlavor, Lld, RustcAbi, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::helenos::opts();
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.linker = Some("i686-helenos-gcc".into());
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m32"]);
    base.rustc_abi = Some(RustcAbi::X86Sse2);

    Target {
        llvm_target: "i686-unknown-helenos".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("IA-32 (i686) HelenOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: Arch::X86,
        options: base,
    }
}
