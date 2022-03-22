use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "freebsd".into(),
        dynamic_linking: true,
        executables: true,
        families: vec!["unix".into()],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        abi_return_struct_as_int: true,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
