// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: func object clause

#![allow(dead_code)]
#![feature(rustc_attrs)]
#![deny(hr_lifetime_in_assoc_type)]

trait Foo<'a> {
    type Item;
}

impl<'a> Foo<'a> for() {
    type Item = ();
}

// Check that appearing in a projection input in the argument is not enough:
#[cfg(func)]
fn func1(_: for<'a> fn(<() as Foo<'a>>::Item) -> &'a i32) {
    //[func]~^ ERROR return type references lifetime `'a`
    //[func]~| WARNING previously accepted
}

// Check that appearing in a projection input in the return still
// causes an error:
#[cfg(func)]
fn func2(_: for<'a> fn() -> <() as Foo<'a>>::Item) {
    //[func]~^ ERROR return type references lifetime `'a`
    //[func]~| WARNING previously accepted
}

#[cfg(object)]
fn object1(_: Box<for<'a> Fn(<() as Foo<'a>>::Item) -> &'a i32>) {
    //[object]~^ ERROR `Output` references lifetime `'a`
    //[object]~| WARNING previously accepted
}

#[cfg(object)]
fn object2(_: Box<for<'a> Fn() -> <() as Foo<'a>>::Item>) {
    //[object]~^ ERROR `Output` references lifetime `'a`
    //[object]~| WARNING previously accepted
}

#[cfg(clause)]
fn clause1<T>() where T: for<'a> Fn(<() as Foo<'a>>::Item) -> &'a i32 {
    //[clause]~^ ERROR `Output` references lifetime `'a`
    //[clause]~| WARNING previously accepted
}

#[cfg(clause)]
fn clause2<T>() where T: for<'a> Fn() -> <() as Foo<'a>>::Item {
    //[clause]~^ ERROR `Output` references lifetime `'a`
    //[clause]~| WARNING previously accepted
}

#[rustc_error]
fn main() { } //[ok]~ ERROR compilation successful
