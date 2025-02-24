use crate::spec::{
    Cc, LinkerFlavor, Lld, SanitizerSet, StackProbeType, Target, TargetMetadata, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::fuchsia::opts();
    base.cpu = "generic".into();
    base.features = "+v8a,+crc,+aes,+sha2,+neon".into();
    base.max_atomic_width = Some(128);
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers = SanitizerSet::ADDRESS
        | SanitizerSet::CFI
        | SanitizerSet::LEAK
        | SanitizerSet::SHADOWCALLSTACK;
    base.supports_xray = true;

    base.add_pre_link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &[
            "--execute-only",
            // Enable the Cortex-A53 errata 843419 mitigation by default
            "--fix-cortex-a53-843419",
        ],
    );

    Target {
        llvm_target: "aarch64-unknown-fuchsia".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Fuchsia".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
