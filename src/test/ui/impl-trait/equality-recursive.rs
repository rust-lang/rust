// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait, specialization)]

trait Foo: Copy + ToString {}

impl<T: Copy + ToString> Foo for T {}

fn sum_to(n: u32) -> impl Foo {
    if n == 0 {
        0
    } else {
        n + sum_to(n - 1)
        //~^ ERROR no implementation for `u32 + impl Foo`
    }
}

fn main() {}