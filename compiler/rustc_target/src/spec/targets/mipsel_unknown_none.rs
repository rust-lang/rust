//! Bare MIPS32r2, little endian, softfloat, O32 calling convention
//!
//! Can be used for MIPS M4K core (e.g. on PIC32MX devices)

use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "mipsel-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("Bare MIPS (LE) softfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            cpu: "mips32r2".into(),
            features: "+mips32r2,+soft-float,+noabicalls".into(),
            max_atomic_width: Some(32),
            linker: Some("rust-lld".into()),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            ..Default::default()
        },
    }
}
