use crate::spec::base::apple::{Arch, TargetAbi, base};
use crate::spec::{SanitizerSet, Target, TargetOptions};

pub(crate) fn target() -> Target {
    // x86_64-apple-ios is a simulator target, even though it isn't declared
    // that way in the target name like the other ones...
    let (opts, llvm_target, arch) = base("ios", Arch::X86_64, TargetAbi::Simulator);
    Target {
        llvm_target,
        metadata: crate::spec::TargetMetadata {
            description: Some("x86_64 Apple iOS Simulator".into()),
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
            supported_sanitizers: SanitizerSet::ADDRESS | SanitizerSet::THREAD,
            ..opts
        },
    }
}
