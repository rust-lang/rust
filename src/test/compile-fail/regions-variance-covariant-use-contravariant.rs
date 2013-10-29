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
// parameter yields an error when used in a covariant way.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

// This is covariant with respect to 'a, meaning that
// Covariant<'foo> <: Covariant<'static> because
// 'foo <= 'static
struct Covariant<'a> {
    f: extern "Rust" fn(&'a int)
}

fn use_<'a>(c: Covariant<'a>) {
    let x = 3;

    // 'b winds up being inferred to 'a because
    // Covariant<'a> <: Covariant<'b> => 'a <= 'b
    //
    // Borrow checker then reports an error because `x` does not
    // have the lifetime 'a.
    collapse(&x, c); //~ ERROR borrowed value does not live long enough


    fn collapse<'b>(x: &'b int, c: Covariant<'b>) { }
}

fn main() {}
