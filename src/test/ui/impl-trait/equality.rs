// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

trait Foo: Copy + ToString {}

impl<T: Copy + ToString> Foo for T {}

fn hide<T: Foo>(x: T) -> impl Foo {
    x
}

fn two(x: bool) -> impl Foo {
    if x {
        return 1_i32;
    }
    0_u32
    //~^ ERROR mismatched types
    //~| expected i32, found u32
}

fn sum_to(n: u32) -> impl Foo {
    if n == 0 {
        0
    } else {
        n + sum_to(n - 1)
        //~^ ERROR cannot add `impl Foo` to `u32`
    }
}

trait Leak: Sized {
    type T;
    fn leak(self) -> Self::T;
}
impl<T> Leak for T {
    default type T = ();
    default fn leak(self) -> Self::T { panic!() }
}
impl Leak for i32 {
    type T = i32;
    fn leak(self) -> i32 { self }
}

fn main() {
}
