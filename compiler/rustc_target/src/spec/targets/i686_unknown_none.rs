use crate::spec::{
    Arch, Cc, LinkerFlavor, Lld, PanicStrategy, RelroLevel, RustcAbi, SanitizerSet, StackProbeType,
    Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        cpu: "x86".into(),
        plt_by_default: false,
        max_atomic_width: Some(32),
        stack_probes: StackProbeType::Inline,
        position_independent_executables: true,
        static_position_independent_executables: true,
        relro_level: RelroLevel::Full,
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        rustc_abi: Some(RustcAbi::Softfloat),
        features: "-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-avx,-avx2,+soft-float".into(),
        supported_sanitizers: SanitizerSet::KCFI | SanitizerSet::KERNELADDRESS,
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        c_int_width: 32,
        ..Default::default()
    };
    Target {
        llvm_target: "i686-unknown-none-elf".into(),
        metadata: TargetMetadata {
            description: Some("Freestanding/bare-metal x86 softfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout:
            "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"
                .into(),
        arch: Arch::X86,
        options: opts,
    }
}
