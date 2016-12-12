// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

trait Foo {
    const NUM: usize;
}

impl Foo for i32 {
    const NUM: usize = 1;
}

const FOO: usize = <i32 as Foo>::NUM;

fn main() {
    assert_eq!(1, FOO);

    match 1 {
        <i32 as Foo>::NUM => {},
        _ => assert!(false)
    }
}
