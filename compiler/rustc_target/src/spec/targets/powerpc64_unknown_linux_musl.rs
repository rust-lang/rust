use rustc_abi::Endian;

use crate::spec::{
    Arch, Cc, LinkerFlavor, Lld, StackProbeType, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::linux_musl::opts();
    base.cpu = "ppc64".into();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;
    base.abi = "elfv2".into();
    base.llvm_abiname = "elfv2".into();

    Target {
        llvm_target: "powerpc64-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some("64-bit PowerPC Linux with musl 1.2.5".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-Fn32-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512".into(),
        arch: Arch::PowerPC64,
        options: TargetOptions { endian: Endian::Big, mcount: "_mcount".into(), ..base },
    }
}
