use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Msvc,
                vec!["/NOLOGO".to_string(),
                     "/NXCOMPAT".to_string()]);

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
        // set VSLANG to 1033 can prevent link.exe from using
        // language packs, and avoid generating Non-UTF-8 error
        // messages if a link error occurred.
        link_env: vec![("VSLANG".to_string(), "1033".to_string())],
        pre_link_args: args,
        crt_static_allows_dylibs: true,
        crt_static_respected: true,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,

        .. Default::default()
    }
}
