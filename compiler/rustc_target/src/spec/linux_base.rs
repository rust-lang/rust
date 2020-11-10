use crate::spec::{LinkArgs, LinkerFlavor, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(
        LinkerFlavor::Gcc,
        vec![
            // We want to be able to strip as much executable code as possible
            // from the linker command line, and this flag indicates to the
            // linker that it can avoid linking in dynamic libraries that don't
            // actually satisfy any symbols up to that point (as with many other
            // resolutions the linker does). This option only applies to all
            // following libraries so we're sure to pass it as one of the first
            // arguments.
            "-Wl,--as-needed".to_string(),
            // Always enable NX protection when it is available
            "-Wl,-z,noexecstack".to_string(),
        ],
    );

    TargetOptions {
        os: "linux".to_string(),
        env: "gnu".to_string(),
        dynamic_linking: true,
        executables: true,
        os_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        pre_link_args: args,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_elf_tls: true,
        crt_static_respected: true,
        ..Default::default()
    }
}
