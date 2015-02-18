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

use std::marker;

struct Covariant<'a> {
    marker: marker::PhantomData<fn(&'a ())>
}

fn use_<'short,'long>(c: Covariant<'long>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {

    // Test whether Covariant<'long> <: Covariant<'short>.  Since
    // 'short <= 'long, this would be true if the Covariant type were
    // contravariant with respect to its parameter 'a.

    let _: Covariant<'short> = c; //~ ERROR mismatched types
}

fn main() {}
