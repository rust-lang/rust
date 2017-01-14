// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-i128_type

fn test1() -> i128 { //~ ERROR 128-bit type is unstable
    0
}

fn test1_2() -> u128 { //~ ERROR 128-bit type is unstable
    0
}

fn test3() {
    let x: i128 = 0; //~ ERROR 128-bit type is unstable
}

fn test3_2() {
    let x: u128 = 0; //~ ERROR 128-bit type is unstable
}

#[repr(u128)]
enum A { //~ ERROR 128-bit type is unstable
    A(u64)
}
