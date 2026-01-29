use rustc_abi::{Align, Endian};

use crate::spec::{
    Arch, Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, RustcAbi, SanitizerSet, StackProbeType,
    Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        cpu: "z10".into(),
        endian: Endian::Big,
        features: "+soft-float,-vector".into(),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        max_atomic_width: Some(128),
        min_global_align: Some(Align::from_bits(16).unwrap()),
        panic_strategy: PanicStrategy::Abort,
        relocation_model: RelocModel::Static,
        rustc_abi: Some(RustcAbi::Softfloat),
        stack_probes: StackProbeType::Inline,
        supported_sanitizers: SanitizerSet::KERNELADDRESS,
        ..Default::default()
    };

    Target {
        llvm_target: "s390x-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("S390x Linux ".into()),
            host_tools: Some(false),
            std: Some(false),
            tier: Some(2),
        },
        arch: Arch::S390x,
        data_layout: "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64".into(),
        options: opts,
        pointer_width: 64,
    }
}
