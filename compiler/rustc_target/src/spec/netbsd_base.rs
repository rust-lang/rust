use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "netbsd".to_string(),
        dynamic_linking: true,
        executables: true,
        families: vec!["unix".to_string()],
        linker_is_gnu: true,
        no_default_libraries: false,
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        use_ctors_section: true,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
