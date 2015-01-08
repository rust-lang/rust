// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

#[derive(PartialEq, PartialOrd, Eq, Ord)]
struct Foo(Box<[u8]>);

pub fn main() {
    let a = Foo(box [0, 1, 2]);
    let b = Foo(box [0, 1, 2]);
    assert!(a == b);
    println!("{}", a != b);
    println!("{}", a < b);
    println!("{}", a <= b);
    println!("{}", a == b);
    println!("{}", a > b);
    println!("{}", a >= b);
}
