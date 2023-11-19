use crate::spec::base::apple::{opts, Arch};
use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let llvm_target = "arm64-apple-ios14.0-macabi";

    let arch = Arch::Arm64_macabi;
    let mut base = opts("ios", arch);
    base.add_pre_link_args(LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-target", llvm_target]);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::LEAK | SanitizerSet::THREAD;

    Target {
        llvm_target: llvm_target.into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a12".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}
