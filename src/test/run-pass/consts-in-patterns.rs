// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]

const FOO: isize = 10;
const BAR: isize = 3;

const fn foo() -> isize { 4 }
const BOO: isize = foo();

pub fn main() {
    let x: isize = 3;
    let y = match x {
        FOO => 1,
        BAR => 2,
        BOO => 4,
        _ => 3
    };
    assert_eq!(y, 2);
}
