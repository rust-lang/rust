// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a type which is invariant with respect to its region
// parameter used in a covariant way yields an error.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

struct Invariant<'a> {
    f: &'static mut &'a int
}

fn use_<'b>(c: Invariant<'b>) {

    // For this assignment to be legal, Invariant<'b> <: Invariant<'static>.
    // Since 'b <= 'static, this would be true if Invariant were covariant
    // with respect to its parameter 'a.

    let _: Invariant<'static> = c; //~ ERROR mismatched types
}

fn main() { }
