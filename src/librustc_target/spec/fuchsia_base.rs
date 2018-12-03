// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LldFlavor, LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Lld(LldFlavor::Ld), vec![
        "--build-id".to_string(),
        "--eh-frame-hdr".to_string(),
        "--hash-style=gnu".to_string(),
        "-z".to_string(), "rodynamic".to_string(),
    ]);

    TargetOptions {
        linker: Some("rust-lld".to_owned()),
        lld_flavor: LldFlavor::Ld,
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        is_like_fuchsia: true,
        linker_is_gnu: true,
        has_rpath: false,
        pre_link_args: pre_link_args,
        pre_link_objects_exe: vec![
            "Scrt1.o".to_string()
        ],
        position_independent_executables: true,
        has_elf_tls: true,
        .. Default::default()
    }
}
