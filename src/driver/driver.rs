// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(rustc, feature(rustc_private))]
#![cfg_attr(rustdoc, feature(rustdoc))]

#[cfg(rustdoc)]
extern crate rustdoc as this;

#[cfg(rustc)]
extern crate rustc_driver as this;

#[cfg_attr(target_os = "nacl", main_link_name = "nacl_main")]
fn main() { this::main() }

#[cfg(target_os = "nacl")]
#[link(name = "ppapi_cpp", kind = "static")]
#[link(name = "ppapi_simple_cpp", kind = "static")]
#[link(name = "ppapi_stub", kind = "static")]
#[link(name = "cli_main", kind = "static")]
#[link(name = "tar", kind = "static")]
#[link(name = "nacl_spawn", kind = "static")]
extern { }
