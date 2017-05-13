// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: angle paren ok elision

#![allow(dead_code)]
#![feature(rustc_attrs)]
#![feature(unboxed_closures)]
#![deny(hr_lifetime_in_assoc_type)]

trait Foo {
    type Item;
}

#[cfg(angle)]
fn angle<T: for<'a> Foo<Item=&'a i32>>() {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
    //[angle]~| WARNING previously accepted
}

#[cfg(angle)]
fn angle1<T>() where T: for<'a> Foo<Item=&'a i32> {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
    //[angle]~| WARNING previously accepted
}

#[cfg(angle)]
fn angle2<T>() where for<'a> T: Foo<Item=&'a i32> {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
    //[angle]~| WARNING previously accepted
}

#[cfg(angle)]
fn angle3(_: &for<'a> Foo<Item=&'a i32>) {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
    //[angle]~| WARNING previously accepted
}

#[cfg(paren)]
fn paren<T: for<'a> Fn() -> &'a i32>() {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
    //[paren]~| WARNING previously accepted
}

#[cfg(paren)]
fn paren1<T>() where T: for<'a> Fn() -> &'a i32 {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
    //[paren]~| WARNING previously accepted
}

#[cfg(paren)]
fn paren2<T>() where for<'a> T: Fn() -> &'a i32 {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
    //[paren]~| WARNING previously accepted
}

#[cfg(paren)]
fn paren3(_: &for<'a> Fn() -> &'a i32) {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
    //[paren]~| WARNING previously accepted
}

#[cfg(elision)]
fn elision<T: Fn() -> &i32>() {
    //[elision]~^ ERROR E0106
}

struct Parameterized<'a> { x: &'a str }

#[cfg(ok)]
fn ok1<T: for<'a> Fn(&Parameterized<'a>) -> &'a i32>() {
}

#[cfg(ok)]
fn ok2<T: for<'a,'b> Fn<(&'b Parameterized<'a>,), Output=&'a i32>>() {
}

#[cfg(ok)]
fn ok3<T>() where for<'a> Parameterized<'a>: Foo<Item=&'a i32> {
}

#[rustc_error]
fn main() { } //[ok]~ ERROR compilation successful
