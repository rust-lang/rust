use crate::abi::Endian;
use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::vxworks_base::opts();
    base.pre_link_args.entry(LinkerFlavor::Gcc).or_default().push("-mspe".into());
    base.pre_link_args.entry(LinkerFlavor::Gcc).or_default().push("--secure-plt".into());
    base.max_atomic_width = Some(32);

    Target {
        llvm_target: "powerpc-unknown-linux-gnuspe".into(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-n32".into(),
        arch: "powerpc".into(),
        options: TargetOptions {
            abi: "spe".into(),
            endian: Endian::Big,
            // feature msync would disable instruction 'fsync' which is not supported by fsl_p1p2
            features: "+secure-plt,+msync".into(),
            ..base
        },
    }
}
