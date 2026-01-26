use crate::spec::{
    Arch, Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, SanitizerSet, StackProbeType, Target,
    TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        // based off the aarch64-unknown-none target at time of addition
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        supported_sanitizers: SanitizerSet::KCFI | SanitizerSet::KERNELADDRESS,
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(128),
        stack_probes: StackProbeType::Inline,
        panic_strategy: PanicStrategy::Abort,
        default_uwtable: true,

        // deviations from aarch64-unknown-none: `+v8a` -> `+v8r`; `+v8r` implies `+neon`
        features: "+v8r,+strict-align".into(),
        ..Default::default()
    };
    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv8-R AArch64, hardfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        // $ clang-21 -S -emit-llvm -target aarch64 -mcpu=cortex-r82 stub.c
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: Arch::AArch64,
        options: opts,
    }
}
