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
    fail_unless!((i32_a == 10));
    fail_unless!((i32_a - 10 == 0));
    fail_unless!((i32_a / 10 == 1));
    fail_unless!((i32_a - 20 == -10));
    fail_unless!((i32_a << 10 == 10240));
    fail_unless!((i32_a << 16 == 655360));
    fail_unless!((i32_a * 16 == 160));
    fail_unless!((i32_a * i32_a * i32_a == 1000));
    fail_unless!((i32_a * i32_a * i32_a * i32_a == 10000));
    fail_unless!((i32_a * i32_a / i32_a * i32_a == 100));
    fail_unless!((i32_a * (i32_a - 1) << 2 + i32_a == 368640));
    let i32_b: int = 0x10101010;
    fail_unless!((i32_b + 1 - 1 == i32_b));
    fail_unless!((i32_b << 1 == i32_b << 1));
    fail_unless!((i32_b >> 1 == i32_b >> 1));
    fail_unless!((i32_b & i32_b << 1 == 0));
    log(debug, i32_b | i32_b << 1);
    fail_unless!((i32_b | i32_b << 1 == 0x30303030));
}
