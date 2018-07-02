// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(trivial_bounds)]
#![allow(unused)]
#![deny(trivial_bounds)]

struct A where i32: Copy; //~ ERROR

trait X<T: Copy> {}

trait Y<T>: Copy {}

trait Z {
    type S: Copy;
}

// Check only the bound the user writes trigger the lint
fn trivial_elaboration<T>() where T: X<i32> + Z<S = i32>, i32: Y<T> {} // OK

fn global_param() where i32: X<()> {} //~ ERROR

// Should only error on the trait bound, not the implicit
// projection bound <i32 as Z>::S == i32.
fn global_projection() where i32: Z<S = i32> {} //~ ERROR

impl A {
    fn new() -> A { A }
}

// Lifetime bounds should be linted as well
fn global_lifetimes() where i32: 'static, &'static str: 'static {}
//~^ ERROR
//~| ERROR

fn local_lifetimes<'a>() where i32: 'a, &'a str: 'a {} // OK

fn global_outlives() where 'static: 'static {} //~ ERROR

// Check that each bound is checked individually
fn mixed_bounds<T: Copy>() where i32: X<T> + Copy {} //~ ERROR

fn main() {}
