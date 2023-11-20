use crate::spec::base::apple::{ios_llvm_target, opts, Arch};
use crate::spec::{FramePointer, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64e;
    let mut base = opts("ios", arch);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::THREAD;

    Target {
        llvm_target: ios_llvm_target(arch).into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a12,+v8.3a,+paca,+pacg".into(),
            max_atomic_width: Some(128),
            forces_embed_bitcode: true,
            frame_pointer: FramePointer::NonLeaf,
            bitcode_llvm_cmdline: "-triple\0\
                arm64e-apple-ios14.1.0\0\
                -emit-obj\0\
                -disable-llvm-passes\0\
                -target-abi\0\
                darwinpcs\0\
                -Os\0"
                .into(),
            ..base
        },
    }
}
