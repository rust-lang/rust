// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// https://github.com/rust-lang/rust/issues/46114

#![feature(const_fn)]

#[derive(Eq, PartialEq)]
struct A { value: u32 }

const fn new(value: u32) -> A {
    A { value }
}

const A_1: A = new(1);
const A_2: A = new(2);

fn main() {
    let a_str = match new(42) {
        A_1 => "A 1",
        A_2 => "A 2",
        _ => "Unknown A",
    };
}
