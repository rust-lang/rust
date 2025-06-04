use crate::spec::{
    Cc, FramePointer, LinkerFlavor, Lld, RelocModel, StackProbeType, TargetOptions, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "cosmo".into(),
        families: cvs!["unix"],
        plt_by_default: false,
        requires_uwtable: false,
        dynamic_linking: false,
        executables: true,
        exe_suffix: ".com.dbg".into(),
        emit_debug_gdb_scripts: false,
        crt_static_default: true,
        crt_static_respected: true,
        allows_weak_linkage: true,
        has_rpath: false,
        has_thread_local: false,
        trap_unreachable: true,
        position_independent_executables: false,
        static_position_independent_executables: false,
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        frame_pointer: FramePointer::Always,
        requires_lto: false,
        eh_frame_header: false,
        no_default_libraries: true,
        max_atomic_width: Some(64),
        linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        stack_probes: StackProbeType::None,
        ..Default::default()
    }
}
