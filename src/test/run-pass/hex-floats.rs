// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



pub fn main() {
    let a = 0x4.0x0123456p-10;
    let b = 0b0100.0b0000_0001_0010_0011_0100_0101_0110p-10;
    let c = -0x1.0xfffp-4_f32;
    assert_eq!(a, b);
    assert_eq!(c, -0.12498474_f32);
}
