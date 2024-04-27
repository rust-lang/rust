use crate::spec::base::apple::{opts, watchos_sim_llvm_target, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::X86_64_sim;
    Target {
        llvm_target: watchos_sim_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions { max_atomic_width: Some(128), ..opts("watchos", arch) },
    }
}
