use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "redox".to_string(),
        env: "relibc".to_string(),
        dynamic_linking: true,
        executables: true,
        os_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_elf_tls: true,
        crt_static_default: true,
        crt_static_respected: true,
        ..Default::default()
    }
}
