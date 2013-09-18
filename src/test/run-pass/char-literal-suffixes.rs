// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let a: int = 'a'i;
    assert_eq!(a, 97i);
    let a: i8 = 'a'i8;
    assert_eq!(a, 97i8);
    let a: i16 = 'a'i16;
    assert_eq!(a, 97i16);
    let a: i32 = 'a'i32;
    assert_eq!(a, 97i32);
    let a: i64 = 'a'i64;
    assert_eq!(a, 97i64);
    let a: uint = 'a'u;
    assert_eq!(a, 97u);
    let a: u8 = 'a'u8;
    assert_eq!(a, 97u8);
    let a: u16 = 'a'u16;
    assert_eq!(a, 97u16);
    let a: u32 = 'a'u32;
    assert_eq!(a, 97u32);
    let a: u64 = 'a'u64;
    assert_eq!(a, 97u64);
}
