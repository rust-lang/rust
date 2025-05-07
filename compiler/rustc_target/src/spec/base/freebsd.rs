use crate::spec::{RelroLevel, TargetOptions, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "freebsd".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        crt_static_respected: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        abi_return_struct_as_int: true,
        default_dwarf_version: 2,
        ..Default::default()
    }
}
