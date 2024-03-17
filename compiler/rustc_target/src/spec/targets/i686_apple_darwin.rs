use crate::spec::base::apple::{macos_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    // ld64 only understands i386 and not i686
    const ARCH: Arch = Arch::I386;
    const OS: &'static str = "macos";
    const ABI: TargetAbi = TargetAbi::Normal;

    let mut base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));
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
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch: ARCH.target_arch(),
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..base },
    }
}
