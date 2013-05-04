// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub enum E64 {
    H64 = 0x7FFF_FFFF_FFFF_FFFF,
    L64 = 0x8000_0000_0000_0000
}
pub enum E32 {
    H32 = 0x7FFF_FFFF,
    L32 = 0x8000_0000
}

pub fn f(e64: E64, e32: E32) -> (bool,bool) {
    (match e64 {
        H64 => true,
        L64 => false
    },
     match e32 {
        H32 => true,
        L32 => false
    })
}

pub fn main() { }
