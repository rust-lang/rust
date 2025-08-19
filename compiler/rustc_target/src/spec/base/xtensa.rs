use rustc_abi::Endian;

use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "none".into(),
        endian: Endian::Little,
        c_int_width: 32,
        linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        executables: true,
        panic_strategy: PanicStrategy::Abort,
        relocation_model: RelocModel::Static,
        emit_debug_gdb_scripts: false,
        atomic_cas: false,
        ..Default::default()
    }
}
