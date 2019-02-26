use crate::spec::{LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut base = super::linux_base::opts();
    // Many of the symbols defined in compiler-rt are also defined in libgcc.
    // Android's linker doesn't like that by default.
    base.pre_link_args
        .get_mut(&LinkerFlavor::Gcc).unwrap().push("-Wl,--allow-multiple-definition".to_string());
    base.is_like_android = true;
    base.position_independent_executables = true;
    base.has_elf_tls = false;
    base.requires_uwtable = true;
    base
}
