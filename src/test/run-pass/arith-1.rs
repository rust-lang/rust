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
    let i32_a: int = 10;
    assert_eq!(i32_a, 10);
    assert_eq!(i32_a - 10, 0);
    assert_eq!(i32_a / 10, 1);
    assert_eq!(i32_a - 20, -10);
    assert_eq!(i32_a << 10, 10240);
    assert_eq!(i32_a << 16, 655360);
    assert_eq!(i32_a * 16, 160);
    assert_eq!(i32_a * i32_a * i32_a, 1000);
    assert_eq!(i32_a * i32_a * i32_a * i32_a, 10000);
    assert_eq!(i32_a * i32_a / i32_a * i32_a, 100);
    assert_eq!(i32_a * (i32_a - 1) << 2 + i32_a, 368640);
    let i32_b: int = 0x10101010;
    assert_eq!(i32_b + 1 - 1, i32_b);
    assert_eq!(i32_b << 1, i32_b << 1);
    assert_eq!(i32_b >> 1, i32_b >> 1);
    assert_eq!(i32_b & i32_b << 1, 0);
    info!(i32_b | i32_b << 1);
    assert_eq!(i32_b | i32_b << 1, 0x30303030);
}
