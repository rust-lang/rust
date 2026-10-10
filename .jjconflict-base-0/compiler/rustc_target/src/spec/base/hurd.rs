use crate::spec::{Os, RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::Hurd,
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        crt_static_respected: true,
        ..Default::default()
    }
}
