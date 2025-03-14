use crate::spec::{Cc, LinkerFlavor, SanitizerSet, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::illumos::opts();
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::Yes), &["-std=c99"]);
    base.max_atomic_width = Some(128);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::CFI;
    base.features = "+v8a".into();

    Target {
        // LLVM does not currently have a separate illumos target,
        // so we still pass Solaris to it
        llvm_target: "aarch64-unknown-solaris2.11".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 illumos".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
