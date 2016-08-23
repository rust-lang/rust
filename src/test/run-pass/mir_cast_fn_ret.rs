// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub extern "C" fn tuple2() -> (u16, u8) {
    (1, 2)
}

pub extern "C" fn tuple3() -> (u8, u8, u8) {
    (1, 2, 3)
}

pub fn test2() -> u8 {
    tuple2().1
}

pub fn test3() -> u8 {
    tuple3().2
}

fn main() {
    assert_eq!(test2(), 2);
    assert_eq!(test3(), 3);
}
