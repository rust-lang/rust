// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a type which is covariant with respect to its region
// parameter yields an error when used in a contravariant way.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

// `S` is contravariant with respect to both parameters.
struct S<'a, 'b> {
    f: &'a int,
    g: &'b int,
}

fn use_<'short,'long>(c: S<'long, 'short>,
                      s: &'short int,
                      l: &'long int,
                      _where:Option<&'short &'long ()>) {

    let _: S<'long, 'short> = c; // OK
    let _: S<'short, 'short> = c; // OK

    // Test whether S<_,'short> <: S<_,'long>.  Since
    // 'short <= 'long, this would be true if the Contravariant type were
    // covariant with respect to its parameter 'a.

    let _: S<'long, 'long> = c; //~ ERROR mismatched types
}

fn main() {}
