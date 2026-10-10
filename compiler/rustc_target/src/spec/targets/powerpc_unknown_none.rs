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
        features: "-hard-float".into(),
        stack_probes: StackProbeType::None,
        emit_debug_gdb_scripts: false,
        frame_pointer: FramePointer::MayOmit,
        endian: Endian::Big,
        llvm_abiname: "eabi".into(),
        ..Default::default()
    };

    Target {
        llvm_target: "powerpc-unknown-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare-metal PPC 32-bit softfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-Fn32-i64:64-n32".into(),
        arch: Arch::PowerPC,
        options: opts,
    }
}
