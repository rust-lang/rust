use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_gnu_base::opts();
    base.cpu = "ppc64le".to_string();
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "powerpc64le-unknown-linux-gnu".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-i64:64-n32:64".to_string(),
        arch: "powerpc64".to_string(),
        options: TargetOptions { mcount: "_mcount".to_string(), ..base },
    }
}
