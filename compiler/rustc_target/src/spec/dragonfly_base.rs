use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "dragonfly".to_string(),
        dynamic_linking: true,
        executables: true,
        os_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
