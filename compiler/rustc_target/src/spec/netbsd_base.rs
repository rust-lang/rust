use crate::spec::{cvs, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "netbsd".into(),
        dynamic_linking: true,
        executables: true,
        families: cvs!["unix"],
        no_default_libraries: false,
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        use_ctors_section: true,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
