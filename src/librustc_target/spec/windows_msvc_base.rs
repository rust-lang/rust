use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let pre_link_args_msvc = vec!["/NOLOGO".to_string(), "/NXCOMPAT".to_string()];
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
        // set VSLANG to 1033 can prevent link.exe from using
        // language packs, and avoid generating Non-UTF-8 error
        // messages if a link error occurred.
        link_env: vec![("VSLANG".to_string(), "1033".to_string())],
        lld_flavor: LldFlavor::Link,
        pre_link_args,
        crt_static_allows_dylibs: true,
        crt_static_respected: true,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        // Currently we don't pass the /NODEFAULTLIB flag to the linker on MSVC
        // as there's been trouble in the past of linking the C++ standard
        // library required by LLVM. This likely needs to happen one day, but
        // in general Windows is also a more controlled environment than
        // Unix, so it's not necessarily as critical that this be implemented.
        //
        // Note that there are also some licensing worries about statically
        // linking some libraries which require a specific agreement, so it may
        // not ever be possible for us to pass this flag.
        no_default_libraries: false,

        ..Default::default()
    }
}
