// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::mem::size_of;

struct t {a: u8, b: i8}
struct u {a: u8, b: i8, c: u8}
struct v {a: u8, b: i8, c: v2, d: u32}
struct v2 {u: char, v: u8}
struct w {a: isize, b: ()}
struct x {a: isize, b: (), c: ()}
struct y {x: isize}

enum e1 {
    a(u8, u32), b(u32), c
}
enum e2 {
    a(u32), b
}

#[repr(C, u8)]
enum e3 {
    a([u16; 0], u8), b
}

pub fn main() {
    assert_eq!(size_of::<u8>(), 1 as usize);
    assert_eq!(size_of::<u32>(), 4 as usize);
    assert_eq!(size_of::<char>(), 4 as usize);
    assert_eq!(size_of::<i8>(), 1 as usize);
    assert_eq!(size_of::<i32>(), 4 as usize);
    assert_eq!(size_of::<t>(), 2 as usize);
    assert_eq!(size_of::<u>(), 3 as usize);
    // Alignment causes padding before the char and the u32.

    assert_eq!(size_of::<v>(),
                16 as usize);
    assert_eq!(size_of::<isize>(), size_of::<usize>());
    assert_eq!(size_of::<w>(), size_of::<isize>());
    assert_eq!(size_of::<x>(), size_of::<isize>());
    assert_eq!(size_of::<isize>(), size_of::<y>());

    // Make sure enum types are the appropriate size, mostly
    // around ensuring alignment is handled properly

    assert_eq!(size_of::<e1>(), 8 as usize);
    assert_eq!(size_of::<e2>(), 8 as usize);
    assert_eq!(size_of::<e3>(), 4 as usize);
}
