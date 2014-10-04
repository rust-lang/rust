// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that regionck suggestions in a provided method of a trait
// don't ICE

trait Foo<'a> {
    fn foo(&'a self);
    fn bar(&self) {
        self.foo();
        //~^ ERROR mismatched types: expected `&'a Self`, found `&Self` (lifetime mismatch)
    }
}

fn main() {}
