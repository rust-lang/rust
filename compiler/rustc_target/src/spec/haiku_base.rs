use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "haiku".to_string(),
        dynamic_linking: true,
        executables: true,
        has_rpath: false,
        os_family: Some("unix".to_string()),
        relro_level: RelroLevel::Full,
        linker_is_gnu: true,
        ..Default::default()
    }
}
