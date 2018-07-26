// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "cdylib"]
#![feature(wasm_custom_section)]
#![deny(warnings)]

extern crate foo;

#[link_section = "foo"]
pub static A: [u8; 2] = [5, 6];

#[link_section = "baz"]
pub static B: [u8; 2] = [7, 8];

#[no_mangle]
pub extern fn foo() {}
