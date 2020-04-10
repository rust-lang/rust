use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let pre_link_args_msvc = vec![
        "/NOLOGO".to_string(),
        "/NXCOMPAT".to_string(),
        "/APPCONTAINER".to_string(),
        "mincore.lib".to_string(),
    ];
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Msvc, pre_link_args_msvc.clone());
    pre_link_args.insert(LinkerFlavor::Lld(LldFlavor::Link), pre_link_args_msvc);

    TargetOptions {
        function_sections: true,
        dynamic_linking: true,
        executables: true,
        dll_prefix: String::new(),
        dll_suffix: ".dll".to_string(),
        exe_suffix: ".exe".to_string(),
        staticlib_prefix: String::new(),
        staticlib_suffix: ".lib".to_string(),
        target_family: Some("windows".to_string()),
        is_like_windows: true,
        is_like_msvc: true,
        pre_link_args,
        crt_static_allows_dylibs: true,
        crt_static_respected: true,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        lld_flavor: LldFlavor::Link,

        ..Default::default()
    }
}
