use crate::spec::{LinkArgs, LinkerFlavor, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(
        LinkerFlavor::Gcc,
        vec![
            // GNU-style linkers will use this to omit linking to libraries
            // which don't actually fulfill any relocations, but only for
            // libraries which follow this flag.  Thus, use it before
            // specifying libraries to link to.
            "-Wl,--as-needed".to_string(),
            // Always enable NX protection when it is available
            "-Wl,-z,noexecstack".to_string(),
        ],
    );

    TargetOptions {
        target_os: "openbsd".to_string(),
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        abi_return_struct_as_int: true,
        pre_link_args: args,
        position_independent_executables: true,
        eliminate_frame_pointer: false, // FIXME 43575
        relro_level: RelroLevel::Full,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
