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

#[wasm_custom_section = "foo"] //~ ERROR: only allowed on consts
static A: [u8; 2] = [1, 2];

#[wasm_custom_section = "foo"] //~ ERROR: only allowed on consts
struct B {}

#[wasm_custom_section = "foo"] //~ ERROR: only allowed on consts
enum C {}

#[wasm_custom_section = "foo"] //~ ERROR: only allowed on consts
impl B {}

#[wasm_custom_section = "foo"] //~ ERROR: only allowed on consts
mod d {}

#[wasm_custom_section = "foo"] //~ ERROR: only allowed on consts
fn main() {}
