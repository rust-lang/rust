// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a type which is contravariant with respect to its region
// parameter compiles successfully when used in a contravariant way.
//
// Note: see compile-fail/variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

struct Contravariant<'a> {
    f: &'a int
}

fn use_<'a>(c: Contravariant<'a>) {
    let x = 3;

    // 'b winds up being inferred to this call.
    // Contravariant<'a> <: Contravariant<'call> is true
    // if 'call <= 'a, which is true, so no error.
    collapse(&x, c);

    fn collapse<'b>(x: &'b int, c: Contravariant<'b>) { }
}

pub fn main() {}
