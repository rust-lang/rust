use rustc_abi::Endian;

use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        llvm_target: "mips".into(),
        metadata: TargetMetadata {
            description: Some("MIPS32r2 BE Baremetal Softfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        arch: "mips".into(),

        options: TargetOptions {
            vendor: "mti".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            endian: Endian::Big,
            cpu: "mips32r2".into(),

            max_atomic_width: Some(32),

            features: "+mips32r2,+soft-float,+noabicalls".into(),
            executables: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            singlethread: true,
            ..Default::default()
        },
    }
}
