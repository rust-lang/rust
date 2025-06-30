use crate::spec::base::apple::{Arch, TargetEnv, base};
use crate::spec::{Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let (opts, llvm_target, arch) = base("macos", Arch::I686, TargetEnv::Normal);
    Target {
        llvm_target,
        metadata: TargetMetadata {
            description: Some("x86 Apple macOS (10.12+, Sierra+)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch,
        options: TargetOptions { mcount: "\u{1}mcount".into(), max_atomic_width: Some(64), ..opts },
    }
}
