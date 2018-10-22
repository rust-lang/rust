// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

// Test that we enforce user-provided type annotations on closures.

fn foo<'a>() {
    // Here `x` is free in the closure sig:
    |x: &'a i32| -> &'static i32 {
        return x; //~ ERROR unsatisfied lifetime constraints
    };
}

fn foo1() {
    // Here `x` is bound in the closure sig:
    |x: &i32| -> &'static i32 {
        return x; //~ ERROR unsatisfied lifetime constraints
    };
}

fn bar<'a>() {
    // Here `x` is free in the closure sig:
    |x: &'a i32, b: fn(&'static i32)| {
        b(x); //~ ERROR unsatisfied lifetime constraints
    };
}

fn bar1() {
    // Here `x` is bound in the closure sig:
    |x: &i32, b: fn(&'static i32)| {
        b(x); //~ ERROR borrowed data escapes outside of closure
    };
}

fn main() { }
