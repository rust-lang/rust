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

trait MyTrait : 'static {}

impl MyTrait for .. {}

fn foo<T:MyTrait>() { }

fn bar<'a>() {
    foo::<&'a ()>(); //~ ERROR does not fulfill the required lifetime
}

fn main() {
}
