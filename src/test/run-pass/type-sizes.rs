// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
use sys::rustrt::size_of;
extern mod std;

pub fn main() {
    assert!((size_of::<u8>() == 1 as uint));
    assert!((size_of::<u32>() == 4 as uint));
    assert!((size_of::<char>() == 4 as uint));
    assert!((size_of::<i8>() == 1 as uint));
    assert!((size_of::<i32>() == 4 as uint));
    assert!((size_of::<{a: u8, b: i8}>() == 2 as uint));
    assert!((size_of::<{a: u8, b: i8, c: u8}>() == 3 as uint));
    // Alignment causes padding before the char and the u32.

    assert!(size_of::<{a: u8, b: i8, c: {u: char, v: u8}, d: u32}>() ==
                16 as uint);
    assert!((size_of::<int>() == size_of::<uint>()));
    assert!((size_of::<{a: int, b: ()}>() == size_of::<int>()));
    assert!((size_of::<{a: int, b: (), c: ()}>() == size_of::<int>()));
    assert!((size_of::<int>() == size_of::<{x: int}>()));
}
