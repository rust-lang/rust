use crate::spec::{LinkArgs, LinkerFlavor, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(
        LinkerFlavor::Gcc,
        vec![
            // Always enable NX protection when it is available
            "-Wl,-z,noexecstack".to_string(),
        ],
    );

    TargetOptions {
        os: "redox".to_string(),
        env: "relibc".to_string(),
        dynamic_linking: true,
        executables: true,
        os_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        pre_link_args: args,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_elf_tls: true,
        crt_static_default: true,
        crt_static_respected: true,
        ..Default::default()
    }
}
