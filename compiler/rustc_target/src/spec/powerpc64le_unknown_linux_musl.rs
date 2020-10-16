use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "ppc64le".to_string();
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "powerpc64le-unknown-linux-musl".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i64:64-n32:64".to_string(),
        arch: "powerpc64".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions { target_mcount: "_mcount".to_string(), ..base },
    }
}
