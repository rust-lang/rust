// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that an invariant region parameter used in a contravariant way
// yields an error.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

struct Invariant<'a> {
    f: &'static mut &'a int
}

fn use_<'short,'long>(c: Invariant<'long>,
                      s: &'short int,
                      l: &'long int,
                      _where:Option<&'short &'long ()>) {

    // Test whether Invariant<'long> <: Invariant<'short>.  Since
    // 'short <= 'long, this would be true if the Invariant type were
    // contravariant with respect to its parameter 'a.

    let _: Invariant<'short> = c; //~ ERROR mismatched types
}

fn main() { }
