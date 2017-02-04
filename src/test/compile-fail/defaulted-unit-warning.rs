// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![allow(unreachable_code)]
#![deny(resolve_trait_on_defaulted_unit)]

trait Deserialize: Sized {
    fn deserialize() -> Result<Self, String>;
}

impl Deserialize for () {
    fn deserialize() -> Result<(), String> {
        Ok(())
    }
}

fn doit() -> Result<(), String> {
    let _ = match Deserialize::deserialize() {
        //~^ ERROR code relies on type
        //~| WARNING previously accepted
        Ok(x) => x,
        Err(e) => return Err(e),
    };
    Ok(())
}

trait ImplementedForUnitButNotNever {}

impl ImplementedForUnitButNotNever for () {}

fn foo<T: ImplementedForUnitButNotNever>(_t: T) {}

fn smeg() {
    let _x = return;
    foo(_x);
    //~^ ERROR code relies on type
    //~| WARNING previously accepted
}

fn main() {
    let _ = doit();
}

