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
        //~^ ERROR the trait bound `u32: std::ops::Add<impl Foo>` is not satisfied
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

trait CheckIfSend: Sized {
    type T: Default;
    fn check(self) -> Self::T { Default::default() }
}
impl<T> CheckIfSend for T {
    default type T = ();
}
impl<T: Send> CheckIfSend for T {
    type T = bool;
}

fn main() {
    let _: u32 = hide(0_u32);
    //~^ ERROR mismatched types
    //~| expected type `u32`
    //~| found type `impl Foo`
    //~| expected u32, found anonymized type

    let _: i32 = Leak::leak(hide(0_i32));
    //~^ ERROR mismatched types
    //~| expected type `i32`
    //~| found type `<impl Foo as Leak>::T`
    //~| expected i32, found associated type

    let _: bool = CheckIfSend::check(hide(0_i32));
    //~^ ERROR mismatched types
    //~| expected type `bool`
    //~| found type `<impl Foo as CheckIfSend>::T`
    //~| expected bool, found associated type

    let mut x = (hide(0_u32), hide(0_i32));
    x = (x.1,
    //~^ ERROR mismatched types
    //~| expected u32, found i32
         x.0);
    //~^ ERROR mismatched types
    //~| expected i32, found u32
}
