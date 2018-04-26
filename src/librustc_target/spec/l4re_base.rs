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
//use std::process::Command;

// Use GCC to locate code for crt* libraries from the host, not from L4Re. Note
// that a few files also come from L4Re, for these, the function shouldn't be
// used. This uses GCC for the location of the file, but GCC is required for L4Re anyway.
//fn get_path_or(filename: &str) -> String {
//    let child = Command::new("gcc")
//        .arg(format!("-print-file-name={}", filename)).output()
//        .expect("Failed to execute GCC");
//    String::from_utf8(child.stdout)
//        .expect("Couldn't read path from GCC").trim().into()
//}

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![]);

    TargetOptions {
        executables: true,
        has_elf_tls: false,
        exe_allocation_crate: None,
        panic_strategy: PanicStrategy::Abort,
        linker: Some("ld".to_string()),
        pre_link_args: args,
        target_family: Some("unix".to_string()),
        .. Default::default()
    }
}
