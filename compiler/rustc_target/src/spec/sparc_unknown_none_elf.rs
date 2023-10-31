use crate::abi::Endian;
use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let options = TargetOptions {
        linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        linker: Some("sparc-elf-gcc".into()),
        endian: Endian::Big,
        cpu: "v7".into(),
        abi: "elf".into(),
        max_atomic_width: Some(32),
        atomic_cas: true,
        panic_strategy: PanicStrategy::Abort,
        relocation_model: RelocModel::Static,
        no_default_libraries: false,
        emit_debug_gdb_scripts: false,
        eh_frame_header: false,
        ..Default::default()
    };
    Target {
        data_layout: "E-m:e-p:32:32-i64:64-f128:64-n32-S64".into(),
        llvm_target: "sparc-unknown-none-elf".into(),
        pointer_width: 32,
        arch: "sparc".into(),
        options,
    }
}
