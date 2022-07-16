use super::apple_sdk_base::{opts, Arch};
use crate::spec::{StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let base = opts("ios", Arch::X86_64_sim);

    // Clang automatically chooses a more specific target based on
    // IPHONEOS_DEPLOYMENT_TARGET.
    // This is required for the simulator target to pick the right
    // MACH-O commands, so we do too.
    let arch = "x86_64";
    let llvm_target = super::apple_base::ios_sim_llvm_target(arch);

    Target {
        llvm_target: llvm_target.into(),
        pointer_width: 64,
        data_layout: "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: TargetOptions {
            max_atomic_width: Some(64),
            // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
            stack_probes: StackProbeType::Call,
            ..base
        },
    }
}
