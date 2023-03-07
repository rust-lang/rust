use super::apple_base::{opts, watchos_sim_llvm_target, Arch};
use crate::spec::{FramePointer, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Arm64_sim;
    Target {
        // Clang automatically chooses a more specific target based on
        // WATCHOS_DEPLOYMENT_TARGET.
        // This is required for the simulator target to pick the right
        // MACH-O commands, so we do too.
        llvm_target: watchos_sim_llvm_target(arch).into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a7".into(),
            max_atomic_width: Some(128),
            forces_embed_bitcode: true,
            frame_pointer: FramePointer::NonLeaf,
            // Taken from a clang build on Xcode 11.4.1.
            // These arguments are not actually invoked - they just have
            // to look right to pass App Store validation.
            bitcode_llvm_cmdline: "-triple\0\
                arm64-apple-watchos5.0-simulator\0\
                -emit-obj\0\
                -disable-llvm-passes\0\
                -target-abi\0\
                darwinpcs\0\
                -Os\0"
                .into(),
            ..opts("watchos", arch)
        },
    }
}
