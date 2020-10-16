use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::vxworks_base::opts();
    base.cpu = "pentium4".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m32".to_string());
    base.stack_probes = true;

    Target {
        llvm_target: "i686-unknown-linux-gnu".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:32-n8:16:32-S128"
            .to_string(),
        arch: "x86".to_string(),
        target_os: "vxworks".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "wrs".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    }
}
