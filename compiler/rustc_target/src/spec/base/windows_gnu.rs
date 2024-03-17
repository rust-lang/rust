use crate::spec::{add_link_args, crt_objects};
use crate::spec::{cvs, Cc, DebuginfoKind, LinkerFlavor, Lld, SplitDebuginfo, TargetOptions};
use crate::spec::{LinkSelfContainedDefault, MaybeLazy};
use std::borrow::Cow;

pub fn opts() -> TargetOptions {
    let pre_link_args = MaybeLazy::lazy(|| {
        let mut pre_link_args = TargetOptions::link_args(
            LinkerFlavor::Gnu(Cc::No, Lld::No),
            &[
                // Enable ASLR
                "--dynamicbase",
                // ASLR will rebase it anyway so leaving that option enabled only leads to confusion
                "--disable-auto-image-base",
            ],
        );
        add_link_args(
            &mut pre_link_args,
            LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            &[
                // Tell GCC to avoid linker plugins, because we are not bundling
                // them with Windows installer, and Rust does its own LTO anyways.
                "-fno-use-linker-plugin",
                "-Wl,--dynamicbase",
                "-Wl,--disable-auto-image-base",
            ],
        );
        pre_link_args
    });

    let late_link_args = MaybeLazy::lazy(|| {
        // Order of `late_link_args*` was found through trial and error to work with various
        // mingw-w64 versions (not tested on the CI). It's expected to change from time to time.
        let mingw_libs = &[
            "-lmsvcrt",
            "-lmingwex",
            "-lmingw32",
            "-lgcc", // alas, mingw* libraries above depend on libgcc
            // mingw's msvcrt is a weird hybrid import library and static library.
            // And it seems that the linker fails to use import symbols from msvcrt
            // that are required from functions in msvcrt in certain cases. For example
            // `_fmode` that is used by an implementation of `__p__fmode` in x86_64.
            // The library is purposely listed twice to fix that.
            //
            // See https://github.com/rust-lang/rust/pull/47483 for some more details.
            "-lmsvcrt",
            // Math functions missing in MSVCRT (they are present in UCRT) require
            // this dependency cycle: `libmingwex.a` -> `libmsvcrt.a` -> `libmingwex.a`.
            "-lmingwex",
            "-luser32",
            "-lkernel32",
        ];
        let mut late_link_args =
            TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), mingw_libs);
        add_link_args(&mut late_link_args, LinkerFlavor::Gnu(Cc::Yes, Lld::No), mingw_libs);
        late_link_args
    });
    // If any of our crates are dynamically linked then we need to use
    // the shared libgcc_s-dw2-1.dll. This is required to support
    // unwinding across DLL boundaries.
    let late_link_args_dynamic = MaybeLazy::lazy(|| {
        let dynamic_unwind_libs = &["-lgcc_s"];
        let mut late_link_args_dynamic =
            TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), dynamic_unwind_libs);
        add_link_args(
            &mut late_link_args_dynamic,
            LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            dynamic_unwind_libs,
        );
        late_link_args_dynamic
    });
    // If all of our crates are statically linked then we can get away
    // with statically linking the libgcc unwinding code. This allows
    // binaries to be redistributed without the libgcc_s-dw2-1.dll
    // dependency, but unfortunately break unwinding across DLL
    // boundaries when unwinding across FFI boundaries.
    let late_link_args_static = MaybeLazy::lazy(|| {
        let static_unwind_libs = &["-lgcc_eh", "-l:libpthread.a"];
        let mut late_link_args_static =
            TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), static_unwind_libs);
        add_link_args(
            &mut late_link_args_static,
            LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            static_unwind_libs,
        );
        late_link_args_static
    });

    TargetOptions {
        os: "windows".into(),
        env: "gnu".into(),
        vendor: "pc".into(),
        // FIXME(#13846) this should be enabled for windows
        function_sections: false,
        linker: Some("gcc".into()),
        dynamic_linking: true,
        dll_tls_export: false,
        dll_prefix: "".into(),
        dll_suffix: ".dll".into(),
        exe_suffix: ".exe".into(),
        families: cvs!["windows"],
        is_like_windows: true,
        allows_weak_linkage: false,
        pre_link_args,
        pre_link_objects: crt_objects::pre_mingw(),
        post_link_objects: crt_objects::post_mingw(),
        pre_link_objects_self_contained: crt_objects::pre_mingw_self_contained(),
        post_link_objects_self_contained: crt_objects::post_mingw_self_contained(),
        link_self_contained: LinkSelfContainedDefault::InferredForMingw,
        late_link_args,
        late_link_args_dynamic,
        late_link_args_static,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        eh_frame_header: false,
        // FIXME(davidtwco): Support Split DWARF on Windows GNU - may require LLVM changes to
        // output DWO, despite using DWARF, doesn't use ELF..
        debuginfo_kind: DebuginfoKind::Pdb,
        supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
        ..Default::default()
    }
}
