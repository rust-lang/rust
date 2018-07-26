// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut base = super::linux_base::opts();

    // Make sure that the linker/gcc really don't pull in anything, including
    // default objects, libs, etc.
    base.pre_link_args_crt.insert(LinkerFlavor::Gcc, Vec::new());
    base.pre_link_args_crt.get_mut(&LinkerFlavor::Gcc).unwrap().push("-nostdlib".to_string());

    // At least when this was tested, the linker would not add the
    // `GNU_EH_FRAME` program header to executables generated, which is required
    // when unwinding to locate the unwinding information. I'm not sure why this
    // argument is *not* necessary for normal builds, but it can't hurt!
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-Wl,--eh-frame-hdr".to_string());

    // There's a whole bunch of circular dependencies when dealing with MUSL
    // unfortunately. To put this in perspective libc is statically linked to
    // liblibc and libunwind is statically linked to libstd:
    //
    // * libcore depends on `fmod` which is in libc (transitively in liblibc).
    //   liblibc, however, depends on libcore.
    // * compiler-rt has personality symbols that depend on libunwind, but
    //   libunwind is in libstd which depends on compiler-rt.
    //
    // Recall that linkers discard libraries and object files as much as
    // possible, and with all the static linking and archives flying around with
    // MUSL the linker is super aggressively stripping out objects. For example
    // the first case has fmod stripped from liblibc (it's in its own object
    // file) so it's not there when libcore needs it. In the second example all
    // the unused symbols from libunwind are stripped (each is in its own object
    // file in libstd) before we end up linking compiler-rt which depends on
    // those symbols.
    //
    // To deal with these circular dependencies we just force the compiler to
    // link everything as a group, not stripping anything out until everything
    // is processed. The linker will still perform a pass to strip out object
    // files but it won't do so until all objects/archives have been processed.
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-Wl,-(".to_string());
    base.post_link_args.insert(LinkerFlavor::Gcc, vec!["-Wl,-)".to_string()]);

    // When generating a statically linked executable there's generally some
    // small setup needed which is listed in these files. These are provided by
    // a musl toolchain and are linked by default by the `musl-gcc` script. Note
    // that `gcc` also does this by default, it just uses some different files.
    //
    // Each target directory for musl has these object files included in it so
    // they'll be included from there.
    base.pre_link_objects_exe_crt.push("crt1.o".to_string());
    base.pre_link_objects_exe_crt.push("crti.o".to_string());
    base.post_link_objects_crt.push("crtn.o".to_string());

    // These targets statically link libc by default
    base.crt_static_default = true;
    // These targets allow the user to choose between static and dynamic linking.
    base.crt_static_respected = true;

    base
}
