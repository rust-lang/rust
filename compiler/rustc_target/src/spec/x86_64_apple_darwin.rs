use crate::spec::TargetOptions;
use crate::spec::{FramePointer, LinkerFlavor, SanitizerSet, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::apple_base::opts("macos");
    base.cpu = "core2".to_string();
    base.max_atomic_width = Some(128); // core2 support cmpxchg16b
    base.frame_pointer = FramePointer::Always;
    base.pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec!["-m64".to_string(), "-arch".to_string(), "x86_64".to_string()],
    );
    base.link_env_remove.extend(super::apple_base::macos_link_env_remove());
    // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
    base.stack_probes = StackProbeType::Call;
    base.supported_sanitizers =
        SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::LEAK | SanitizerSet::THREAD;

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
