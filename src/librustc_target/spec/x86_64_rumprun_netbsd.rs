use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::netbsd_base::opts();
    base.cpu = "x86-64".to_string();
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());
    base.linker = Some("x86_64-rumprun-netbsd-gcc".to_string());
    base.max_atomic_width = Some(64);

    base.dynamic_linking = false;
    base.has_rpath = false;
    base.position_independent_executables = false;
    base.disable_redzone = true;
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "x86_64-rumprun-netbsd".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        target_os: "netbsd".to_string(),
        target_env: String::new(),
        target_vendor: "rumprun".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            target_mcount: "__mcount".to_string(),
            .. base
        },
    })
}
