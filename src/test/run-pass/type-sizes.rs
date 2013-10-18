// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use std::mem::size_of;

struct t {a: u8, b: i8}
struct u {a: u8, b: i8, c: u8}
struct v {a: u8, b: i8, c: v2, d: u32}
struct v2 {u: char, v: u8}
struct w {a: int, b: ()}
struct x {a: int, b: (), c: ()}
struct y {x: int}

pub fn main() {
    assert_eq!(size_of::<u8>(), 1 as uint);
    assert_eq!(size_of::<u32>(), 4 as uint);
    assert_eq!(size_of::<char>(), 4 as uint);
    assert_eq!(size_of::<i8>(), 1 as uint);
    assert_eq!(size_of::<i32>(), 4 as uint);
    assert_eq!(size_of::<t>(), 2 as uint);
    assert_eq!(size_of::<u>(), 3 as uint);
    // Alignment causes padding before the char and the u32.

    assert!(size_of::<v>() ==
                16 as uint);
    assert_eq!(size_of::<int>(), size_of::<uint>());
    assert_eq!(size_of::<w>(), size_of::<int>());
    assert_eq!(size_of::<x>(), size_of::<int>());
    assert_eq!(size_of::<int>(), size_of::<y>());
}
