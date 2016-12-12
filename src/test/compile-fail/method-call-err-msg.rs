// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that parameter cardinality or missing method error gets span exactly.

pub struct Foo;
impl Foo {
    fn zero(self) -> Foo { self }
    //~^ NOTE defined here
    fn one(self, _: isize) -> Foo { self }
    //~^ NOTE defined here
    fn two(self, _: isize, _: isize) -> Foo { self }
    //~^ NOTE defined here
}

fn main() {
    let x = Foo;
    x.zero(0)   //~ ERROR this function takes 0 parameters but 1 parameter was supplied
     //~^ NOTE expected 0 parameters
     .one()     //~ ERROR this function takes 1 parameter but 0 parameters were supplied
     //~^ NOTE expected 1 parameter
     .two(0);   //~ ERROR this function takes 2 parameters but 1 parameter was supplied
     //~^ NOTE expected 2 parameters

    let y = Foo;
    y.zero()
     .take()    //~ ERROR no method named `take` found for type `Foo` in the current scope
     //~^ NOTE the method `take` exists but the following trait bounds were not satisfied
     .one(0);
}
