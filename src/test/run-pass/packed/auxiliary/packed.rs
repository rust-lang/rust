// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(packed)]
pub struct P1S5 {
    a: u8,
    b: u32
}

#[repr(packed(2))]
pub struct P2S6 {
    a: u8,
    b: u32,
    c: u8
}

#[repr(C, packed(2))]
pub struct P2CS8 {
    a: u8,
    b: u32,
    c: u8
}
