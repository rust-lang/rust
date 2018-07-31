// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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
    args.insert(LinkerFlavor::Gcc, vec![
        "-Wl,-Bstatic".to_string(),
        "-Wl,--no-dynamic-linker".to_string(),
        "-Wl,--gc-sections".to_string(),
        "-Wl,--as-needed".to_string(),
    ]);

    TargetOptions {
        exe_allocation_crate: None,
        executables: true,
        has_elf_tls: true,
        linker_is_gnu: true,
        no_default_libraries: false,
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: false,
        pre_link_args: args,
        relocation_model: "static".to_string(),
        target_family: Some("unix".to_string()),
        tls_model: "local-exec".to_string(),
        .. Default::default()
    }
}
