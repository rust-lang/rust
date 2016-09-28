// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

union U {
    principal: u8,
}

impl U {
    fn calculate(&self) {}
}

fn main() {
    let u = U { principle: 0 };
    //~^ ERROR union `U` has no field named `principle`
    //~| NOTE field does not exist - did you mean `principal`?
    let w = u.principial; //~ ERROR no field `principial` on type `U`
                          //~^ did you mean `principal`?

    let y = u.calculate; //~ ERROR attempted to take value of method `calculate` on type `U`
                         //~^ HELP maybe a `()` to call it is missing?
}
