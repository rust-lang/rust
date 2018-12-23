// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:custom_derive_plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(custom_derive_plugin)]

trait TotalSum {
    fn total_sum(&self) -> isize;
}

impl TotalSum for isize {
    fn total_sum(&self) -> isize {
        *self
    }
}

struct Seven;

impl TotalSum for Seven {
    fn total_sum(&self) -> isize {
        7
    }
}

#[derive_TotalSum]
struct Foo {
    seven: Seven,
    bar: Bar,
    baz: isize,
}

#[derive_TotalSum]
struct Bar {
    quux: isize,
    bleh: isize,
}


pub fn main() {
    let v = Foo {
        seven: Seven,
        bar: Bar {
            quux: 9,
            bleh: 3,
        },
        baz: 80,
    };
    assert_eq!(v.total_sum(), 99);
}
