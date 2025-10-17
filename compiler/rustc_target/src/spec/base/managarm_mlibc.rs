use crate::spec::{RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "managarm".into(),
        env: "mlibc".into(),
        dynamic_linking: true,
        executables: true,
        families: cvs!["unix"],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        crt_static_respected: true,
        ..Default::default()
    }
}
