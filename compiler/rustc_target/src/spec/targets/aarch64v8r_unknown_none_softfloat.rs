use crate::spec::{
    Abi, Arch, Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, SanitizerSet, StackProbeType,
    Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        abi: Abi::SoftFloat,
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(128),
        supported_sanitizers: SanitizerSet::KCFI | SanitizerSet::KERNELADDRESS,
        stack_probes: StackProbeType::Inline,
        panic_strategy: PanicStrategy::Abort,
        default_uwtable: true,

        // deviations from aarch64-unknown-none: `+v8a` -> `+v8r`
        features: "+v8r,+strict-align,-neon".into(),
        ..Default::default()
    };
    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv8-R AArch64, softfloat".into()),
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
