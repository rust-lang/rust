use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::apple_base::opts("macos");
    base.cpu = "yonah".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m32".to_string()]);
    base.link_env_remove.extend(super::apple_base::macos_link_env_remove());
    base.stack_probes = true;
    base.eliminate_frame_pointer = false;

    // Clang automatically chooses a more specific target based on
    // MACOSX_DEPLOYMENT_TARGET.  To enable cross-language LTO to work
    // correctly, we do too.
    let arch = "i686";
    let llvm_target = super::apple_base::macos_llvm_target(&arch);

    Target {
        llvm_target,
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:128-n8:16:32-S128"
            .to_string(),
        arch: "x86".to_string(),
        options: TargetOptions { target_mcount: "\u{1}mcount".to_string(), ..base },
    }
}
