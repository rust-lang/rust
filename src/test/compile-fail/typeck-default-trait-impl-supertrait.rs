// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when a `..` impl applies, we also check that any
// supertrait conditions are met.

#![feature(optin_builtin_traits)]

trait NotImplemented { }

trait MyTrait : NotImplemented {}

impl MyTrait for .. {}

fn foo<T:MyTrait>() { bar::<T>() }

fn bar<T:NotImplemented>() { }

fn main() {
    foo::<i32>(); //~ ERROR the trait `NotImplemented` is not implemented for the type `i32`
    bar::<i64>(); //~ ERROR the trait `NotImplemented` is not implemented for the type `i64`
}
