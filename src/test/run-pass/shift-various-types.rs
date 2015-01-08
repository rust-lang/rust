// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can do shifts by any integral type.

struct Panolpy {
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    isize: isize,

    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    usize: usize,
}

fn foo(p: &Panolpy) {
    assert_eq!(22_i32 >> p.i8, 11_i32);
    assert_eq!(22_i32 >> p.i16, 11_i32);
    assert_eq!(22_i32 >> p.i32, 11_i32);
    assert_eq!(22_i32 >> p.i64, 11_i32);
    assert_eq!(22_i32 >> p.isize, 11_i32);

    assert_eq!(22_i32 >> p.u8, 11_i32);
    assert_eq!(22_i32 >> p.u16, 11_i32);
    assert_eq!(22_i32 >> p.u32, 11_i32);
    assert_eq!(22_i32 >> p.u64, 11_i32);
    assert_eq!(22_i32 >> p.usize, 11_i32);
}

fn main() {
    let p = Panolpy {
        i8: 1,
        i16: 1,
        i32: 1,
        i64: 1,
        isize: 1,

        u8: 1,
        u16: 1,
        u32: 1,
        u64: 1,
        usize: 1,
    };
    foo(&p)
}
