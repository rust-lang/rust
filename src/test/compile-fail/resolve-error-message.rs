// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause. Specifically, test that we cannot nest
// quantification in constraints (to be clear, there is no reason this should not
// we're testing we don't crash or do something stupid).

mod a { use c::*; mod b { mod c { struct Foo; } } }
//~^ unresolved import `c::*`. Maybe a missing `extern crate c`, or do you mean

fn main() { }
