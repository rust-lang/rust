// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We need to opt inot the `!` feature in order to trigger the
// requirement that this is testing.
#![feature(never_type)]

#![allow(unused)]

trait Deserialize: Sized {
    fn deserialize() -> Result<Self, String>;
}

impl Deserialize for () {
    fn deserialize() -> Result<(), String> {
        Ok(())
    }
}

trait ImplementedForUnitButNotNever {}

impl ImplementedForUnitButNotNever for () {}

fn foo<T: ImplementedForUnitButNotNever>(_t: T) {}
//~^ NOTE required by `foo`

fn smeg() {
    let _x = return;
    foo(_x);
    //~^ ERROR the trait bound
    //~| NOTE the trait `ImplementedForUnitButNotNever` is not implemented
    //~| NOTE the trait is implemented for `()`
}

fn main() {
    smeg();
}

