use crate::spec::{RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        crt_static_respected: true,
        dynamic_linking: true,
        executables: true,
        families: cvs!["unix"],
        has_rpath: true,
        has_thread_local: false,
        linker: Some("qcc".into()),
        os: "nto".into(),
        position_independent_executables: true,
        static_position_independent_executables: true,
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
