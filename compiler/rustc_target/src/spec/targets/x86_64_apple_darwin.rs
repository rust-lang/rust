use crate::spec::base::apple::{Arch, TargetAbi, base};
use crate::spec::{SanitizerSet, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let (opts, llvm_target, arch) = base("macos", Arch::X86_64, TargetAbi::Normal);
    Target {
        llvm_target,
        metadata: TargetMetadata {
            description: Some("x86_64 Apple macOS (10.12+, Sierra+)".into()),
            tier: Some(1),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch,
        options: TargetOptions {
            mcount: "\u{1}mcount".into(),
            max_atomic_width: Some(128), // penryn+ supports cmpxchg16b
            supported_sanitizers: SanitizerSet::ADDRESS
                | SanitizerSet::CFI
                | SanitizerSet::LEAK
                | SanitizerSet::THREAD,
            supports_xray: true,
            ..opts
        },
    }
}
