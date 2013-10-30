// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::size_of;

#[repr(i8)]
enum Ei8 {
    Ai8 = 0,
    Bi8 = 1
}

#[repr(u8)]
enum Eu8 {
    Au8 = 0,
    Bu8 = 1
}

#[repr(i16)]
enum Ei16 {
    Ai16 = 0,
    Bi16 = 1
}

#[repr(u16)]
enum Eu16 {
    Au16 = 0,
    Bu16 = 1
}

#[repr(i32)]
enum Ei32 {
    Ai32 = 0,
    Bi32 = 1
}

#[repr(u32)]
enum Eu32 {
    Au32 = 0,
    Bu32 = 1
}

#[repr(i64)]
enum Ei64 {
    Ai64 = 0,
    Bi64 = 1
}

#[repr(u64)]
enum Eu64 {
    Au64 = 0,
    Bu64 = 1
}

#[repr(int)]
enum Eint {
    Aint = 0,
    Bint = 1
}

#[repr(uint)]
enum Euint {
    Auint = 0,
    Buint = 1
}

pub fn main() {
    assert_eq!(size_of::<Ei8>(), 1);
    assert_eq!(size_of::<Eu8>(), 1);
    assert_eq!(size_of::<Ei16>(), 2);
    assert_eq!(size_of::<Eu16>(), 2);
    assert_eq!(size_of::<Ei32>(), 4);
    assert_eq!(size_of::<Eu32>(), 4);
    assert_eq!(size_of::<Ei64>(), 8);
    assert_eq!(size_of::<Eu64>(), 8);
    assert_eq!(size_of::<Eint>(), size_of::<int>());
    assert_eq!(size_of::<Euint>(), size_of::<uint>());
}
