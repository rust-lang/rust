use crate::spec::base::apple::{base, Arch, TargetAbi};
use crate::spec::{SanitizerSet, Target, TargetOptions};

pub(crate) fn target() -> Target {
    let (opts, llvm_target, arch) = base("ios", Arch::X86_64, TargetAbi::MacCatalyst);
    Target {
        llvm_target,
        metadata: crate::spec::TargetMetadata {
            description: Some("Apple Catalyst on x86_64".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch,
        options: TargetOptions {
            max_atomic_width: Some(128),
            supported_sanitizers: SanitizerSet::ADDRESS | SanitizerSet::LEAK | SanitizerSet::THREAD,
            ..opts
        },
    }
}
