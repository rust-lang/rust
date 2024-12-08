// Generic x86-64 target for bare-metal code - Floating point disabled
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.

use crate::spec::{
    Cc, CodeModel, LinkerFlavor, Lld, PanicStrategy, RelroLevel, SanitizerSet, StackProbeType,
    Target, TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        cpu: "x86-64".into(),
        plt_by_default: false,
        max_atomic_width: Some(64),
        stack_probes: StackProbeType::Inline,
        position_independent_executables: true,
        static_position_independent_executables: true,
        relro_level: RelroLevel::Full,
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        features: "-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-avx,-avx2,+soft-float".into(),
        supported_sanitizers: SanitizerSet::KCFI | SanitizerSet::KERNELADDRESS,
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        code_model: Some(CodeModel::Kernel),
        ..Default::default()
    };
    Target {
        llvm_target: "x86_64-unknown-none-elf".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Freestanding/bare-metal x86_64 softfloat".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: opts,
    }
}
