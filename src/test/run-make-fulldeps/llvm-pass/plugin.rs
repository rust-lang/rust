// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin_registrar, rustc_private)]
#![crate_type = "dylib"]
#![crate_name = "some_plugin"]

extern crate rustc;
extern crate rustc_plugin;

#[link(name = "llvm-function-pass", kind = "static")]
#[link(name = "llvm-module-pass", kind = "static")]
extern {}

use rustc_plugin::registry::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_llvm_pass("some-llvm-function-pass");
    reg.register_llvm_pass("some-llvm-module-pass");
}
