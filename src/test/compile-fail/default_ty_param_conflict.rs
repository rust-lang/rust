// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

use std::fmt::Debug;

// Example from the RFC
fn foo<F:Default=usize>() -> F { F::default() }
//~^ NOTE: a default was defined here...

fn bar<B:Debug=isize>(b: B) { println!("{:?}", b); }
//~^ NOTE: a second default was defined here...

fn main() {
    // Here, F is instantiated with $0=uint
    let x = foo();
    //~^ ERROR: mismatched types
    //~| expected type `usize`
    //~| found type `isize`
    //~| NOTE: conflicting type parameter defaults `usize` and `isize`
    //~| NOTE: conflicting type parameter defaults `usize` and `isize`
    //~| NOTE: ...that was applied to an unconstrained type variable here

    // Here, B is instantiated with $1=uint, and constraint $0 <: $1 is added.
    bar(x);
    //~^ NOTE: ...that also applies to the same type variable here
}
