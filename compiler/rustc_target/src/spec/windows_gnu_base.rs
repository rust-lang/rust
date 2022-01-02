use crate::spec::crt_objects::{self, CrtObjectsFallback};
use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            // Tell GCC to avoid linker plugins, because we are not bundling
            // them with Windows installer, and Rust does its own LTO anyways.
            "-fno-use-linker-plugin".to_string(),
            // Enable ASLR
            "-Wl,--dynamicbase".to_string(),
            // ASLR will rebase it anyway so leaving that option enabled only leads to confusion
            "-Wl,--disable-auto-image-base".to_string(),
        ],
    );

    let mut late_link_args_dynamic = LinkArgs::new();
    let mut late_link_args_static = LinkArgs::new();
    let dynamic_unwind_libs = vec![
        // If any of our crates are dynamically linked then we need to use
        // the shared libgcc_s-dw2-1.dll. This is required to support
        // unwinding across DLL boundaries.
        "-l:libunwind.dll.a".to_string(),
    ];
    late_link_args_dynamic.insert(LinkerFlavor::Gcc, dynamic_unwind_libs.clone());
    late_link_args_dynamic.insert(LinkerFlavor::Lld(LldFlavor::Ld), dynamic_unwind_libs);
    let static_unwind_libs = vec![
        // If all of our crates are statically linked then we can get away
        // with statically linking the libgcc unwinding code. This allows
        // binaries to be redistributed without the libgcc_s-dw2-1.dll
        // dependency, but unfortunately break unwinding across DLL
        // boundaries when unwinding across FFI boundaries.
        "-l:libunwind.a".to_string(),
    ];
    late_link_args_static.insert(LinkerFlavor::Gcc, static_unwind_libs.clone());
    late_link_args_static.insert(LinkerFlavor::Lld(LldFlavor::Ld), static_unwind_libs);

    TargetOptions {
        os: "windows".to_string(),
        env: "gnu".to_string(),
        vendor: "pc".to_string(),
        // FIXME(#13846) this should be enabled for windows
        function_sections: true,
        no_default_libraries: false,
        linker: Some("gcc".to_string()),
        dynamic_linking: true,
        executables: true,
        dll_prefix: String::new(),
        dll_suffix: ".dll".to_string(),
        exe_suffix: ".exe".to_string(),
        families: vec!["windows".to_string()],
        is_like_windows: true,
        allows_weak_linkage: false,
        pre_link_args,
        pre_link_objects: crt_objects::pre_mingw(),
        post_link_objects: crt_objects::post_mingw(),
        pre_link_objects_fallback: crt_objects::pre_mingw_fallback(),
        post_link_objects_fallback: crt_objects::post_mingw_fallback(),
        crt_objects_fallback: Some(CrtObjectsFallback::Mingw),
        late_link_args_dynamic,
        late_link_args_static,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        eh_frame_header: false,

        ..Default::default()
    }
}
