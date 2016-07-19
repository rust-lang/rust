// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
//aux-build:default_ty_param_cross_crate_crate.rs

#![feature(default_type_parameter_fallback)]

extern crate default_param_test;

use default_param_test::{Foo, bleh};

fn meh<X, B=bool>(x: Foo<X, B>) {}
//~^ NOTE: a default was defined here...

fn main() {
    let foo = bleh();
    //~^ NOTE: ...that also applies to the same type variable here

    meh(foo);
    //~^ ERROR: mismatched types
    //~| NOTE: conflicting type parameter defaults `bool` and `char`
    //~| NOTE: conflicting type parameter defaults `bool` and `char`
    //~| a second default is defined on `default_param_test::bleh`
    //~| NOTE:  ...that was applied to an unconstrained type variable here
    //~| expected type `bool`
    //~| found type `char`
}
