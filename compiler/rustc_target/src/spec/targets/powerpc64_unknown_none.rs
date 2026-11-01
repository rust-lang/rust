use rustc_abi::Endian;
use crate::spec::{
    Arch, Cc, FramePointer, LinkerFlavor, Lld, PanicStrategy, RelocModel, StackProbeType,
    Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        relocation_model: RelocModel::Static,
        panic_strategy: PanicStrategy::Abort,
        features: "-mma,-altivec,-vsx,-hard-float".into(),
        stack_probes: StackProbeType::None,
        emit_debug_gdb_scripts: false,
        frame_pointer: FramePointer::MayOmit,
        endian: Endian::Big,
        llvm_abiname: "elfv1".into(),
        ..Default::default()
    };

    Target {
        llvm_target: "powerpc64-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("Bare-metal PPC64 softfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "E-m:e-Fi64-i64:64-i128:128-n32:64".into(),
        arch: Arch::PowerPC64,
        options: opts,
    }
}
