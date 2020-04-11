use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            // Tell GCC to avoid linker plugins, because we are not bundling
            // them with Windows installer, and Rust does its own LTO anyways.
            "-fno-use-linker-plugin".to_string(),
            // Always enable DEP (NX bit) when it is available
            "-Wl,--nxcompat".to_string(),
            // Do not use the standard system startup files or libraries when linking
            "-nostdlib".to_string(),
        ],
    );

    let mut late_link_args = LinkArgs::new();
    let mut late_link_args_dynamic = LinkArgs::new();
    let mut late_link_args_static = LinkArgs::new();
    late_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            "-lmingwex".to_string(),
            "-lmingw32".to_string(),
            "-lmsvcrt".to_string(),
            // mingw's msvcrt is a weird hybrid import library and static library.
            // And it seems that the linker fails to use import symbols from msvcrt
            // that are required from functions in msvcrt in certain cases. For example
            // `_fmode` that is used by an implementation of `__p__fmode` in x86_64.
            // Listing the library twice seems to fix that, and seems to also be done
            // by mingw's gcc (Though not sure if it's done on purpose, or by mistake).
            //
            // See https://github.com/rust-lang/rust/pull/47483
            "-lmsvcrt".to_string(),
            "-luser32".to_string(),
            "-lkernel32".to_string(),
        ],
    );
    late_link_args_dynamic.insert(
        LinkerFlavor::Gcc,
        vec![
            // If any of our crates are dynamically linked then we need to use
            // the shared libgcc_s-dw2-1.dll. This is required to support
            // unwinding across DLL boundaries.
            "-lgcc_s".to_string(),
            "-lgcc".to_string(),
            "-lkernel32".to_string(),
        ],
    );
    late_link_args_static.insert(
        LinkerFlavor::Gcc,
        vec![
            // If all of our crates are statically linked then we can get away
            // with statically linking the libgcc unwinding code. This allows
            // binaries to be redistributed without the libgcc_s-dw2-1.dll
            // dependency, but unfortunately break unwinding across DLL
            // boundaries when unwinding across FFI boundaries.
            "-lgcc_eh".to_string(),
            "-l:libpthread.a".to_string(),
            "-lgcc".to_string(),
            // libpthread depends on libmsvcrt, so we need to link it *again*.
            "-lmsvcrt".to_string(),
            "-lkernel32".to_string(),
        ],
    );

    TargetOptions {
        // FIXME(#13846) this should be enabled for windows
        function_sections: false,
        linker: Some("gcc".to_string()),
        dynamic_linking: true,
        executables: true,
        dll_prefix: String::new(),
        dll_suffix: ".dll".to_string(),
        exe_suffix: ".exe".to_string(),
        staticlib_prefix: "lib".to_string(),
        staticlib_suffix: ".a".to_string(),
        target_family: Some("windows".to_string()),
        is_like_windows: true,
        allows_weak_linkage: false,
        pre_link_args,
        pre_link_objects_exe: vec![
            "crt2.o".to_string(),    // mingw C runtime initialization for executables
            "rsbegin.o".to_string(), // Rust compiler runtime initialization, see rsbegin.rs
        ],
        pre_link_objects_dll: vec![
            "dllcrt2.o".to_string(), // mingw C runtime initialization for dlls
            "rsbegin.o".to_string(),
        ],
        late_link_args,
        late_link_args_dynamic,
        late_link_args_static,
        post_link_objects: vec!["rsend.o".to_string()],
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,

        ..Default::default()
    }
}
