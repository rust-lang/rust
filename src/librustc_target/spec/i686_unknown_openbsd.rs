use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::openbsd_base::opts();
    base.cpu = "pentium4".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m32".to_string());
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-fuse-ld=lld".to_string());
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "i686-unknown-openbsd".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128".to_string(),
        arch: "x86".to_string(),
        target_os: "openbsd".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
