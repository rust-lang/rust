use crate::spec::base::apple::{macos_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, MaybeLazy, SanitizerSet};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::X86_64;
    const OS: &'static str = "macos";
    const ABI: TargetAbi = TargetAbi::Normal;

    let mut base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));
    base.max_atomic_width = Some(128); // penryn+ supports cmpxchg16b
    base.frame_pointer = FramePointer::Always;
    base.pre_link_args =
        TargetOptions::link_args(LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-m64"]);
    base.supported_sanitizers =
        SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::LEAK | SanitizerSet::THREAD;

    Target {
        // Clang automatically chooses a more specific target based on
        // MACOSX_DEPLOYMENT_TARGET. To enable cross-language LTO to work
        // correctly, we do too.
        llvm_target: MaybeLazy::lazy(|| macos_llvm_target(ARCH)),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions { mcount: "\u{1}mcount".into(), ..base },
    }
}
