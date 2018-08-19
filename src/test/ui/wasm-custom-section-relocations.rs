// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// only-wasm32

#[link_section = "test"]
pub static A: &[u8] = &[1]; //~ ERROR: no extra levels of indirection

#[link_section = "test"]
pub static B: [u8; 3] = [1, 2, 3];

#[link_section = "test"]
pub static C: usize = 3;

#[link_section = "test"]
pub static D: &usize = &C; //~ ERROR: no extra levels of indirection
