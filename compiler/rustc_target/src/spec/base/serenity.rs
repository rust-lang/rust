use crate::spec::{FramePointer, Os, PanicStrategy, RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Serenity,
        dynamic_linking: true,
        frame_pointer: FramePointer::Always,
        families: cvs!["unix"],
        default_dwarf_version: 5,
        has_rpath: true,
        no_default_libraries: false,
        position_independent_executables: true,
        static_position_independent_executables: false,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        panic_strategy: PanicStrategy::Abort,
        crt_static_default: false,
        crt_static_respected: false,
        ..Default::default()
    }
}
