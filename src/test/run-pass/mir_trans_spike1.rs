// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A simple spike test for MIR version of trans.

#![feature(rustc_attrs)]

#[rustc_mir]
fn sum(x: i32, y: i32) -> i32 {
    x + y
}

fn main() {
    let x = sum(22, 44);
    assert_eq!(x, 66);
    println!("sum()={:?}", x);
}
