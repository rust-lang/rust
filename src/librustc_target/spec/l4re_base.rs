// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkArgs, LinkerFlavor, PanicStrategy, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![]);

    TargetOptions {
        executables: true,
        has_elf_tls: false,
        exe_allocation_crate: None,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("l4-bender".to_string()),
        pre_link_args: args,
        target_family: Some("unix".to_string()),
        .. Default::default()
    }
}
