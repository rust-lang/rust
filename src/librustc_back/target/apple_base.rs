// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;

use target::TargetOptions;

pub fn opts() -> TargetOptions {
    // ELF TLS is only available in OSX 10.7+. If you try to compile for 10.6
    // either the linker will complain if it is used or the binary will end up
    // segfaulting at runtime when run on 10.6. Rust by default supports OSX
    // 10.7+, but there is a standard environment variable,
    // MACOSX_DEPLOYMENT_TARGET, which is used to signal targeting older
    // versions of OSX. For example compiling on 10.10 with
    // MACOSX_DEPLOYMENT_TARGET set to 10.6 will cause the linker to generate
    // warnings about the usage of ELF TLS.
    //
    // Here we detect what version is being requested, defaulting to 10.7. ELF
    // TLS is flagged as enabled if it looks to be supported.
    let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET").ok();
    let version = deployment_target.as_ref().and_then(|s| {
        let mut i = s.splitn(2, ".");
        i.next().and_then(|a| i.next().map(|b| (a, b)))
    }).and_then(|(a, b)| {
        a.parse::<u32>().and_then(|a| b.parse::<u32>().map(|b| (a, b))).ok()
    }).unwrap_or((10, 7));

    TargetOptions {
        // OSX has -dead_strip, which doesn't rely on function_sections
        function_sections: false,
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        is_like_osx: true,
        has_rpath: true,
        dll_prefix: "lib".to_string(),
        dll_suffix: ".dylib".to_string(),
        archive_format: "bsd".to_string(),
        pre_link_args: Vec::new(),
        exe_allocation_crate: super::maybe_jemalloc(),
        has_elf_tls: version >= (10, 7),
        .. Default::default()
    }
}
