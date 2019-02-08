use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::solaris_base::opts();
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m64".to_string()]);
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "x86_64-pc-solaris".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        target_os: "solaris".to_string(),
        target_env: String::new(),
        target_vendor: "sun".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
