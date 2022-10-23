use super::apple_base::{opts, Arch};
use crate::spec::{StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::I386;
    let base = opts("ios", arch);
    let llvm_target = super::apple_base::ios_sim_llvm_target(arch);

    Target {
        llvm_target: llvm_target.into(),
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: TargetOptions {
            max_atomic_width: Some(64),
            stack_probes: StackProbeType::X86,
            ..base
        },
    }
}
