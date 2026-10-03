// Hyperlight Guest target for AArch64

use crate::spec::{Arch, SanitizerSet, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        features: "+v8.1a,+strict-align,+neon,+fp-armv8".into(),
        direct_access_external_data: Some(true),

        supported_sanitizers: SanitizerSet::KCFI
            | SanitizerSet::KERNELADDRESS
            | SanitizerSet::KERNELHWADDRESS,
        max_atomic_width: Some(128),
        default_uwtable: true,
        supports_xray: true,
        ..base::hyperlight::opts()
    };
    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Hyperlight guest".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: Arch::AArch64,
        options: opts,
    }
}
