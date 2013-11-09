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
// parameter is successful when used in a covariant way.
//
// Note: see compile-fail/variance-regions-*.rs for the tests that
// check that the variance inference works in the first place.

// This is covariant with respect to 'a, meaning that
// Covariant<'foo> <: Covariant<'static> because
// 'foo <= 'static
struct Covariant<'a> {
    f: extern "Rust" fn(&'a int)
}

fn use_<'a>(c: Covariant<'a>) {
    // OK Because Covariant<'a> <: Covariant<'static> iff 'a <= 'static
    let _: Covariant<'static> = c;
}

pub fn main() {}
