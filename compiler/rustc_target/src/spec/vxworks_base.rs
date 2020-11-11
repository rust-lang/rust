use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};

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
        ],
    );

    TargetOptions {
        os: "vxworks".to_string(),
        env: "gnu".to_string(),
        vendor: "wrs".to_string(),
        linker: Some("wr-c++".to_string()),
        exe_suffix: ".vxe".to_string(),
        dynamic_linking: true,
        executables: true,
        os_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        pre_link_args: args,
        position_independent_executables: false,
        has_elf_tls: true,
        crt_static_default: true,
        crt_static_respected: true,
        crt_static_allows_dylibs: true,
        // VxWorks needs to implement this to support profiling
        mcount: "_mcount".to_string(),
        ..Default::default()
    }
}
