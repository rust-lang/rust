use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        target_os: "haiku".to_string(),
        dynamic_linking: true,
        executables: true,
        has_rpath: false,
        target_family: Some("unix".to_string()),
        relro_level: RelroLevel::Full,
        linker_is_gnu: true,
        ..Default::default()
    }
}
