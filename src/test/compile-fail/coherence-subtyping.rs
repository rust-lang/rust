// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that two distinct impls which match subtypes of one another
// yield coherence errors (or not) depending on the variance.

trait Contravariant {
    fn foo(&self) { }
}

impl Contravariant for for<'a,'b> fn(&'a u8, &'b u8) {
    //~^ ERROR E0119
}

impl Contravariant for for<'a> fn(&'a u8, &'a u8) {
}

///////////////////////////////////////////////////////////////////////////

trait Covariant {
    fn foo(&self) { }
}

impl Covariant for for<'a,'b> fn(&'a u8, &'b u8) {
    //~^ ERROR E0119
}

impl Covariant for for<'a> fn(&'a u8, &'a u8) {
}

///////////////////////////////////////////////////////////////////////////

trait Invariant {
    fn foo(&self) -> Self { }
}

impl Invariant for for<'a,'b> fn(&'a u8, &'b u8) {
}

impl Invariant for for<'a> fn(&'a u8, &'a u8) {
}

fn main() { }
