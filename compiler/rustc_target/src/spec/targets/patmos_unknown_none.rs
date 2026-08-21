use rustc_abi::Endian;

use crate::spec::{Abi, Arch, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let options = TargetOptions {
        endian: Endian::Big,
        os: crate::spec::Os::None,
        env: crate::spec::Env::Other("unknown".into()),
        vendor: "unknown".into(),
        cpu: "generic".into(),
        linker: std::env::var("CUSTOM_LINKER").ok().map(|s| s.into()),
        abi: Abi::Ilp32,
        max_atomic_width: Some(0),
        panic_strategy: PanicStrategy::Abort,
        relocation_model: RelocModel::Static,
        executables: true,
        //
        no_default_libraries: true,
        linker_flavor: crate::spec::LinkerFlavor::Gnu(crate::spec::Cc::No, crate::spec::Lld::Yes),
        position_independent_executables: false,
        //
        eh_frame_header: false,
        emit_debug_gdb_scripts: false,
        ..Default::default()
    };

    Target {
        llvm_target: "patmos-unknown-unknown-elf".into(),
        metadata: TargetMetadata {
            description: Some("Patmos bare-metal".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "E-S32-p:32:32:32-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f64:32:32-a0:0:32-s0:32:32-v64:32:32-v128:32:32-n32".into(),
        arch: Arch::Patmos,
        options,
    }
}
