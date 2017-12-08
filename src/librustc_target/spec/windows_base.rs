// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Gcc, vec![
            // And here, we see obscure linker flags #45. On windows, it has been
            // found to be necessary to have this flag to compile liblibc.
            //
            // First a bit of background. On Windows, the file format is not ELF,
            // but COFF (at least according to LLVM). COFF doesn't officially allow
            // for section names over 8 characters, apparently. Our metadata
            // section, ".note.rustc", you'll note is over 8 characters.
            //
            // On more recent versions of gcc on mingw, apparently the section name
            // is *not* truncated, but rather stored elsewhere in a separate lookup
            // table. On older versions of gcc, they apparently always truncated th
            // section names (at least in some cases). Truncating the section name
            // actually creates "invalid" objects [1] [2], but only for some
            // introspection tools, not in terms of whether it can be loaded.
            //
            // Long story short, passing this flag forces the linker to *not*
            // truncate section names (so we can find the metadata section after
            // it's compiled). The real kicker is that rust compiled just fine on
            // windows for quite a long time *without* this flag, so I have no idea
            // why it suddenly started failing for liblibc. Regardless, we
            // definitely don't want section name truncation, so we're keeping this
            // flag for windows.
            //
            // [1] - https://sourceware.org/bugzilla/show_bug.cgi?id=13130
            // [2] - https://code.google.com/p/go/issues/detail?id=2139
            "-Wl,--enable-long-section-names".to_string(),

            // Tell GCC to avoid linker plugins, because we are not bundling
            // them with Windows installer, and Rust does its own LTO anyways.
            "-fno-use-linker-plugin".to_string(),

            // Always enable DEP (NX bit) when it is available
            "-Wl,--nxcompat".to_string(),

            // Do not use the standard system startup files or libraries when linking
            "-nostdlib".to_string(),
        ]);

    let mut late_link_args = LinkArgs::new();
    late_link_args.insert(LinkerFlavor::Gcc, vec![
        "-lmingwex".to_string(),
        "-lmingw32".to_string(),
        "-lgcc".to_string(), // alas, mingw* libraries above depend on libgcc
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
    ]);

    TargetOptions {
        // FIXME(#13846) this should be enabled for windows
        function_sections: false,
        linker: Some("gcc".to_string()),
        dynamic_linking: true,
        executables: true,
        dll_prefix: "".to_string(),
        dll_suffix: ".dll".to_string(),
        exe_suffix: ".exe".to_string(),
        staticlib_prefix: "".to_string(),
        staticlib_suffix: ".lib".to_string(),
        no_default_libraries: true,
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
        post_link_objects: vec![
            "rsend.o".to_string()
        ],
        custom_unwind_resume: true,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,

        .. Default::default()
    }
}
