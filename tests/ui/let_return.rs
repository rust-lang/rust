// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]
#![warn(clippy::let_and_return)]

fn test() -> i32 {
    let _y = 0; // no warning
    let x = 5;
    x
}

fn test_inner() -> i32 {
    if true {
        let x = 5;
        x
    } else {
        0
    }
}

fn test_nowarn_1() -> i32 {
    let mut x = 5;
    x += 1;
    x
}

fn test_nowarn_2() -> i32 {
    let x = 5;
    x + 1
}

fn test_nowarn_3() -> (i32, i32) {
    // this should technically warn, but we do not compare complex patterns
    let (x, y) = (5, 9);
    (x, y)
}

fn test_nowarn_4() -> i32 {
    // this should technically warn, but not b/c of clippy::let_and_return, but b/c of useless type
    let x: i32 = 5;
    x
}

fn main() {}
