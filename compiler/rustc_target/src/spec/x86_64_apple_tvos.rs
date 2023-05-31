use super::apple_base::{opts, Arch};
use crate::spec::{StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::X86_64_sim;
    Target {
        llvm_target: "x86_64-apple-tvos".into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-f80:128-n8:16:32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::X86,
            ..opts("tvos", arch)
        },
    }
}
