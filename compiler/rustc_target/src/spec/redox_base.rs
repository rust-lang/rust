use crate::spec::{cvs, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "redox".into(),
        env: "relibc".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        crt_static_default: true,
        crt_static_respected: true,
        ..Default::default()
    }
}
