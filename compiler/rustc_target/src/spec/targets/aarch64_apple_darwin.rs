use crate::spec::base::apple::{macos_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{FramePointer, MaybeLazy, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::Arm64;
    const OS: &'static str = "macos";
    const ABI: TargetAbi = TargetAbi::Normal;

    let mut base = opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)));
    base.cpu = "apple-m1".into();
    base.max_atomic_width = Some(128);

    // FIXME: The leak sanitizer currently fails the tests, see #88132.
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::CFI | SanitizerSet::THREAD;

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
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            mcount: "\u{1}mcount".into(),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}
