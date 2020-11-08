use crate::spec::{LinkerFlavor, RelroLevel, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_base::opts();
    base.cpu = "ppc64".to_string();
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());
    base.max_atomic_width = Some(64);

    // ld.so in at least RHEL6 on ppc64 has a bug related to BIND_NOW, so only enable partial RELRO
    // for now. https://github.com/rust-lang/rust/pull/43170#issuecomment-315411474
    base.relro_level = RelroLevel::Partial;

    Target {
        llvm_target: "powerpc64-unknown-linux-gnu".to_string(),
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64".to_string(),
        arch: "powerpc64".to_string(),
        options: TargetOptions {
            target_endian: "big".to_string(),
            target_mcount: "_mcount".to_string(),
            ..base
        },
    }
}
