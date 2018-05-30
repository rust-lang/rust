// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(where_clauses_object_safety)]

trait Trait {}

trait X {
    fn foo(&self) where Self: Trait; //~ ERROR the trait `X` cannot be made into an object
    //~^ WARN this was previously accepted by the compiler but is being phased out
}

impl X for () {
    fn foo(&self) {}
}

impl Trait for dyn X {}

pub fn main() {
    // Check that this does not segfault.
    <X as X>::foo(&());
}
