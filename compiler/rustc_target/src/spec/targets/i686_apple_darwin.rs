use crate::spec::base::apple::{macos_llvm_target, opts, Arch};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, Target, TargetOptions};

pub fn target() -> Target {
    // ld64 only understands i386 and not i686
    let arch = Arch::I386;
    let mut base = opts("macos", arch);
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-m32"]);
    base.frame_pointer = FramePointer::Always;

    Target {
        // Clang automatically chooses a more specific target based on
        // MACOSX_DEPLOYMENT_TARGET. To enable cross-language LTO to work
        // correctly, we do too.
        //
        // While ld64 doesn't understand i686, LLVM does.
        llvm_target: macos_llvm_target(Arch::I686).into(),
        description: None,
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch: arch.target_arch(),
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..base },
    }
}
