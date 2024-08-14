use crate::spec::base::apple::{mac_catalyst_llvm_target, opts, Arch, TargetAbi};
use crate::spec::{SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::X86_64;
    let mut base = opts("ios", arch, TargetAbi::MacCatalyst);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::LEAK | SanitizerSet::THREAD;

    Target {
        llvm_target: mac_catalyst_llvm_target(arch).into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Apple Catalyst on x86_64".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions { max_atomic_width: Some(128), ..base },
    }
}
