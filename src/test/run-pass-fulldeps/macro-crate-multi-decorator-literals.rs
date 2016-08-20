// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro_crate_test.rs
// ignore-stage1

#![feature(plugin, custom_attribute, attr_literals)]
#![plugin(macro_crate_test)]

#[macro_use]
#[no_link]
extern crate macro_crate_test;

// The `caller(name, args...)` attribute emits a new nullary function named
// `name` that calls the annotated function with `args`. As an example, consider
// the following:
//
//     #[caller(simple, 1, "hello", 3.14)]
//     fn f(num: isize, string: &'static str, float: f32) -> (isize, &'static str, float) {
//         (num, string, float)
//     }
//
// This results in a function named `simple` that calls `f(1, "hello", 3.14)`.
// As a result, the expression `simple()` evaluates to `(1, "helllo", 3.14)`.

#[caller(simple, 1, "hello", 3.14)]
#[caller(simple1, 2, "bye", 6.28)]
#[caller(simple2, 3, "hi", 1.01)]
fn f(num: isize, string: &'static str, float: f32) -> (isize, &'static str, f32) {
    (num, string, float)
}

#[caller(complex, true, 10)]
#[caller(complex1, false, 15)]
#[caller(complex2, true, 20)]
fn g(emit: bool, num: i32) -> Option<i32> {
    match emit {
        true => Some(num),
        false => None
    }
}

fn main() {
    assert_eq!(simple(), (1, "hello", 3.14));
    assert_eq!(simple1(), (2, "bye", 6.28));
    assert_eq!(simple2(), (3, "hi", 1.01));

    assert_eq!(complex(), Some(10));
    assert_eq!(complex1(), None);
    assert_eq!(complex2(), Some(20));
}
