use crate::abi::Endian;
use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::vxworks_base::opts();
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-mspe".to_string());
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("--secure-plt".to_string());
    base.max_atomic_width = Some(32);

    Target {
        llvm_target: "powerpc-unknown-linux-gnuspe".to_string(),
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-i64:64-n32".to_string(),
        arch: "powerpc".to_string(),
        options: TargetOptions {
            endian: Endian::Big,
            // feature msync would disable instruction 'fsync' which is not supported by fsl_p1p2
            features: "+secure-plt,+msync".to_string(),
            ..base
        },
    }
}
