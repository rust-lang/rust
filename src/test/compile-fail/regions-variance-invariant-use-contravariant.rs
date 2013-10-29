// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a covariant region parameter used in a covariant position
// yields an error.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

struct Invariant<'a> {
    f: &'static mut &'a int
}

fn use_<'a>(c: Invariant<'a>) {
    let x = 3;

    // 'b winds up being inferred to 'a, because that is the
    // only way that Invariant<'a> <: Invariant<'b>, and hence
    // we get an error in the borrow checker because &x cannot
    // live that long
    collapse(&x, c); //~ ERROR borrowed value does not live long enough

    fn collapse<'b>(x: &'b int, c: Invariant<'b>) { }
}

fn main() { }
