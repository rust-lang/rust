use crate::abi::Endian;
use crate::spec::{Cc, LinkerFlavor, Lld, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::freebsd_base::opts();
    base.cpu = "ppc64".into();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m64"]);
    base.max_atomic_width = Some(64);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc64-unknown-freebsd".into(),
        pointer_width: 64,
        data_layout: "E-m:e-Fn32-i64:64-n32:64".into(),
        arch: "powerpc64".into(),
        options: TargetOptions { endian: Endian::Big, mcount: "_mcount".into(), ..base },
    }
}
