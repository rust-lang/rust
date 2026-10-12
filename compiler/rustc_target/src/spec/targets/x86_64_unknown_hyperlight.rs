// Hyperlight Guest target for x86_64

use crate::spec::{Arch, RelroLevel, SanitizerSet, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        features: "-mmx,+sse,+sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-avx,-avx2,-soft-float".into(),

        cpu: "x86-64".into(),
        plt_by_default: false,
        max_atomic_width: Some(64),
        relro_level: RelroLevel::Full,
        supported_sanitizers: SanitizerSet::KCFI | SanitizerSet::KERNELADDRESS,
        ..base::hyperlight::opts()
    };
    Target {
        llvm_target: "x86_64-unknown-none-elf".into(),
        metadata: TargetMetadata {
            description: Some("x86_64 Hyperlight guest".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: Arch::X86_64,
        options: opts,
    }
}
