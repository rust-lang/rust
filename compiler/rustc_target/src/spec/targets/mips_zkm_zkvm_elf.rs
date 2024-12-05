use crate::abi::Endian;
use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        llvm_target: "mips".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ZKM's zero-knowledge Virtual Machine (MIPS32r2 ISA)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        arch: "mips".into(),

        options: TargetOptions {
            os: "zkvm".into(),
            vendor: "zkm".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            endian: Endian::Big,
            cpu: "mips32r2".into(),

            // Some crates (*cough* crossbeam) assume you have 64 bit
            // atomics if the target name is not in a hardcoded list.
            // Since zkvm is singlethreaded and all operations are
            // atomic, I guess we can just say we support 64-bit
            // atomics.
            max_atomic_width: Some(64),
            atomic_cas: true,

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
