use crate::spec::base::apple::{ios_llvm_target, opts, Arch};
use crate::spec::{FramePointer, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64e;
    let mut base = opts("ios", arch);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::THREAD;

    Target {
        // Clang automatically chooses a more specific target based on
        // IPHONEOS_DEPLOYMENT_TARGET.
        // This is required for the target to pick the right
        // MACH-O commands, so we do too.
        llvm_target: ios_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a12,+v8.3a,+pauth".into(),
            max_atomic_width: Some(128),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}
