// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[cfg(any(target_arch = "x86", target_arch = "arm"))]
fn target() {
    assert_eq!(-1000 as uint >> 3_usize, 536870787_usize);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn target() {
    assert_eq!(-1000 as uint >> 3_usize, 2305843009213693827_usize);
}

fn general() {
    let mut a: int = 1;
    let mut b: int = 2;
    a ^= b;
    b ^= a;
    a = a ^ b;
    println!("{}", a);
    println!("{}", b);
    assert_eq!(b, 1);
    assert_eq!(a, 2);
    assert_eq!(!0xf0_isize & 0xff, 0xf);
    assert_eq!(0xf0_isize | 0xf, 0xff);
    assert_eq!(0xf_isize << 4, 0xf0);
    assert_eq!(0xf0_isize >> 4, 0xf);
    assert_eq!(-16 >> 2, -4);
    assert_eq!(0b1010_1010_isize | 0b0101_0101, 0xff);
}

pub fn main() {
    general();
    target();
}
