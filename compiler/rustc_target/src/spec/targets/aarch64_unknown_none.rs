// Generic AArch64 target for bare-metal code - Floating point enabled
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.
//
// For example, `-C target-cpu=cortex-a53`.

use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, SanitizerSet, StackProbeType, Target,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        // Enable the Cortex-A53 errata 843419 mitigation by default
        pre_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), &[
            "--fix-cortex-a53-843419",
        ]),
        features: "+v8a,+strict-align,+neon,+fp-armv8".into(),
        supported_sanitizers: SanitizerSet::KCFI | SanitizerSet::KERNELADDRESS,
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(128),
        stack_probes: StackProbeType::Inline,
        panic_strategy: PanicStrategy::Abort,
        ..Default::default()
    };
    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Bare ARM64, hardfloat".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: opts,
    }
}
