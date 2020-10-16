use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::vxworks_base::opts();
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-mspe".to_string());
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("--secure-plt".to_string());
    base.max_atomic_width = Some(32);

    Target {
        llvm_target: "powerpc-unknown-linux-gnuspe".to_string(),
        target_endian: "big".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-p:32:32-i64:64-n32".to_string(),
        arch: "powerpc".to_string(),
        target_os: "vxworks".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "wrs".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            // feature msync would disable instruction 'fsync' which is not supported by fsl_p1p2
            features: "+secure-plt,+msync".to_string(),
            ..base
        },
    }
}
