// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: sig local structure ok elision

#![allow(dead_code)]
#![feature(rustc_attrs)]
#![feature(unboxed_closures)]
#![deny(hr_lifetime_in_assoc_type)]

trait Foo {
    type Item;
}

#[cfg(sig)]
fn sig1(_: for<'a> fn() -> &'a i32) {
    //[sig]~^ ERROR return type references lifetime `'a`
    //[sig]~| WARNING previously accepted
}

#[cfg(sig)]
fn sig2(_: for<'a, 'b> fn(&'b i32) -> &'a i32) {
    //[sig]~^ ERROR return type references lifetime `'a`
    //[sig]~| WARNING previously accepted
}

#[cfg(local)]
fn local1() {
    let _: for<'a> fn() -> &'a i32 = loop { };
    //[local]~^ ERROR return type references lifetime `'a`
    //[local]~| WARNING previously accepted
}

#[cfg(structure)]
struct Struct1 {
    x: for<'a> fn() -> &'a i32
    //[structure]~^ ERROR return type references lifetime `'a`
    //[structure]~| WARNING previously accepted
}

#[cfg(elision)]
fn elision(_: fn() -> &i32) {
    //[elision]~^ ERROR E0106
}

struct Parameterized<'a> { x: &'a str }

#[cfg(ok)]
fn ok1(_: &for<'a> Fn(&Parameterized<'a>) -> &'a i32) {
}

#[cfg(ok)]
fn ok2(_: &for<'a,'b> Fn<(&'b Parameterized<'a>,), Output=&'a i32>) {
}

#[rustc_error]
fn main() { } //[ok]~ ERROR compilation successful
