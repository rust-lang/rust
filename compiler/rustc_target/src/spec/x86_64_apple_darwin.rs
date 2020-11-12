use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::apple_base::opts("macos");
    base.cpu = "core2".to_string();
    base.max_atomic_width = Some(128); // core2 support cmpxchg16b
    base.eliminate_frame_pointer = false;
    base.pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec!["-m64".to_string(), "-arch".to_string(), "x86_64".to_string()],
    );
    base.link_env_remove.extend(super::apple_base::macos_link_env_remove());
    base.stack_probes = true;

    // Clang automatically chooses a more specific target based on
    // MACOSX_DEPLOYMENT_TARGET.  To enable cross-language LTO to work
    // correctly, we do too.
    let arch = "x86_64";
    let llvm_target = super::apple_base::macos_llvm_target(&arch);

    Target {
        llvm_target,
        pointer_width: 64,
        data_layout: "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: arch.to_string(),
        options: TargetOptions { mcount: "\u{1}mcount".to_string(), ..base },
    }
}
