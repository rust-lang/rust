use crate::spec::base::apple::{macos_llvm_target, opts, Arch, TargetAbi};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, Target, TargetOptions};

pub(crate) fn target() -> Target {
    let arch = Arch::I686;
    let mut base = opts("macos", arch, TargetAbi::Normal);
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-m32"]);
    base.frame_pointer = FramePointer::Always;

    Target {
        // Clang automatically chooses a more specific target based on
        // MACOSX_DEPLOYMENT_TARGET. To enable cross-language LTO to work
        // correctly, we do too.
        llvm_target: macos_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("32-bit macOS (10.12+, Sierra+)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch: arch.target_arch(),
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..base },
    }
}
