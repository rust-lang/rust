// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(wasm_custom_section)]

#[wasm_custom_section = "foo"]
const A: u8 = 0; //~ ERROR: must be an array of bytes

#[wasm_custom_section = "foo"]
const B: &[u8] = &[0]; //~ ERROR: must be an array of bytes

#[wasm_custom_section = "foo"]
const C: &[u8; 1] = &[0]; //~ ERROR: must be an array of bytes

fn main() {}
