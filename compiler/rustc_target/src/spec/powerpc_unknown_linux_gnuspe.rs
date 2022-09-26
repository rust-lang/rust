use crate::abi::Endian;
use crate::spec::{LinkerFlavor, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_gnu_base::opts();
    base.add_pre_link_args(LinkerFlavor::Gcc, &["-mspe"]);
    base.max_atomic_width = Some(32);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc-unknown-linux-gnuspe".into(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-n32".into(),
        arch: "powerpc".into(),
        options: TargetOptions {
            abi: "spe".into(),
            endian: Endian::Big,
            mcount: "_mcount".into(),
            ..base
        },
    }
}
