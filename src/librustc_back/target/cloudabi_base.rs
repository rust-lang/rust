// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use LinkerFlavor;
use target::{LinkArgs, TargetOptions, RelroLevel};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![
        "-Wl,-Bstatic".to_string(),
        "-Wl,--no-dynamic-linker".to_string(),
        "-Wl,--eh-frame-hdr".to_string(),
        "-Wl,--gc-sections".to_string(),
    ]);

    TargetOptions {
        executables: true,
        target_family: Some("unix".to_string()),
        linker_is_gnu: true,
        pre_link_args: args,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        exe_allocation_crate: super::maybe_jemalloc(),
        .. Default::default()
    }
}
