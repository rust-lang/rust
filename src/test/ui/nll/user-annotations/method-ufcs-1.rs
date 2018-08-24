// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unit test for the "user substitutions" that are annotated on each
// node.

#![feature(nll)]

trait Bazoom<T>: Sized {
    fn method<U>(self, arg: T, arg2: U) { }
}

impl<T, U> Bazoom<U> for T {
}

fn annot_underscore() {
    let a = 22;
    let b = 44;
    let c = 66;
    <_ as Bazoom<_>>::method::<_>(&a, b, c); // OK
}

fn annot_reference_any_lifetime() {
    let a = 22;
    let b = 44;
    let c = 66;
    <&u32 as Bazoom<_>>::method(&a, b, c); // OK
}

fn annot_reference_static_lifetime() {
    let a = 22;
    let b = 44;
    let c = 66;
    let x = <&'static u32 as Bazoom<_>>::method;
    x(&a, b, c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let a = 22;
    let b = 44;
    let c = 66;
    <&'a u32 as Bazoom<_>>::method(&a, b, c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(a: &'a u32) {
    let b = 44;
    let c = 66;
    <&'a u32 as Bazoom<_>>::method(&a, b, c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let a = 22;
    let b = 44;
    let _closure = || {
        let c = 66;
        <&'a u32 as Bazoom<_>>::method(&a, b, c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(a: &'a u32) {
    let b = 44;
    let c = 66;
    let _closure = || {
        <&'a u32 as Bazoom<_>>::method(&a, b, c);
    };
}

fn main() { }
