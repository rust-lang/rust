use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::apple_base::opts();
    base.cpu = "apple-a12".to_string();
    base.max_atomic_width = Some(128);
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-arch".to_string(), "arm64".to_string()]);

    base.link_env_remove.extend(super::apple_base::macos_link_env_remove());

    // Clang automatically chooses a more specific target based on
    // MACOSX_DEPLOYMENT_TARGET.  To enable cross-language LTO to work
    // correctly, we do too.
    let arch = "aarch64";
    let llvm_target = super::apple_base::macos_llvm_target(&arch);

    Target {
        llvm_target,
        target_endian: "little".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".to_string(),
        arch: arch.to_string(),
        target_os: "macos".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions { target_mcount: "\u{1}mcount".to_string(), ..base },
    }
}
