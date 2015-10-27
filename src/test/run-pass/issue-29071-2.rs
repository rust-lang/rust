// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn t1() -> u32 {
    let x;
    x = if true { [1, 2, 3] } else { [2, 3, 4] }[0];
    x
}

fn t2() -> [u32; 1] {
    if true { [1, 2, 3]; } else { [2, 3, 4]; }
    [0]
}

fn t3() -> u32 {
    let x;
    x = if true { i1 as F } else { i2 as F }();
    x
}

fn t4() -> () {
    if true { i1 as F; } else { i2 as F; }
    ()
}

type F = fn() -> u32;
fn i1() -> u32 { 1 }
fn i2() -> u32 { 2 }

fn main() {
    assert_eq!(t1(), 1);
    assert_eq!(t3(), 1);
}
