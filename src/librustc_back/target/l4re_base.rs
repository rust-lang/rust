// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use PanicStrategy;
use LinkerFlavor;
use target::{LinkArgs, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Ld, vec![
            "-nostdlib".to_string(),
    ]);

    TargetOptions {
        executables: true,
        has_elf_tls: false,
        exe_allocation_crate: Some("alloc_system".to_string()),
        panic_strategy: PanicStrategy::Abort,
        linker: "ld".to_string(),
        pre_link_args: pre_link_args,
        target_family: Some("unix".to_string()),
        .. Default::default()
    }
}
