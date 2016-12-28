// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::TargetOptions;
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        function_sections: true,
        linker: "link.exe".to_string(),
        // When taking a look at the value of this `ar` field, one might expect
        // `lib.exe` to be the value here! The `lib.exe` program is the default
        // tool for managing `.lib` archives on Windows, but unfortunately the
        // compiler cannot use it.
        //
        // To recap, we use `ar` here to manage rlibs (which are just archives).
        // LLVM does not expose bindings for modifying archives so we have to
        // invoke this utility for write operations (e.g. deleting files, adding
        // files, etc). Normally archives only have object files within them,
        // but the compiler also uses archives for storing metadata and
        // compressed bytecode, so we don't exactly fall within "normal use
        // cases".
        //
        // MSVC's `lib.exe` tool by default will choke when adding a non-object
        // file to an archive, which we do on a regular basis, making it
        // inoperable for us. Luckily, however, LLVM has already rewritten `ar`
        // in the form of `llvm-ar` which is built by default when we build
        // LLVM. This tool, unlike `lib.exe`, works just fine with non-object
        // files, so we use it instead.
        //
        // Note that there's a few caveats associated with this:
        //
        // * This still requires that the *linker* (the consumer of rlibs) will
        //   ignore non-object files. Thankfully `link.exe` on Windows does
        //   indeed ignore non-object files in archives.
        // * This requires `llvm-ar.exe` to be distributed with the compiler
        //   itself, but we already make sure of this elsewhere.
        //
        // Perhaps one day we won't even need this tool at all and we'll just be
        // able to make library calls into LLVM!
        ar: "llvm-ar.exe".to_string(),
        dynamic_linking: true,
        executables: true,
        dll_prefix: "".to_string(),
        dll_suffix: ".dll".to_string(),
        exe_suffix: ".exe".to_string(),
        staticlib_prefix: "".to_string(),
        staticlib_suffix: ".lib".to_string(),
        target_family: Some("windows".to_string()),
        is_like_windows: true,
        is_like_msvc: true,
        pre_link_args: vec![
            "/NOLOGO".to_string(),
            "/NXCOMPAT".to_string(),
        ],
        exe_allocation_crate: "alloc_system".to_string(),

        .. Default::default()
    }
}
