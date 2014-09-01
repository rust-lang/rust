// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:regions-bounded-method-type-parameters-cross-crate-lib.rs

// Check explicit region bounds on methods in the cross crate case.

extern crate "regions-bounded-method-type-parameters-cross-crate-lib" as lib;

use lib::Inv;
use lib::MaybeOwned;
use lib::IntoMaybeOwned;

fn call_into_maybe_owned<'a,F:IntoMaybeOwned<'a>>(f: F) {
    // Exercise a code path I found to be buggy. We were not encoding
    // the region parameters from the receiver correctly on trait
    // methods.
    f.into_maybe_owned();
}

fn call_bigger_region<'a, 'b>(a: Inv<'a>, b: Inv<'b>) {
    // Here the value provided for 'y is 'b, and hence 'b:'a does not hold.
    a.bigger_region(b) //~ ERROR cannot infer
}

fn main() { }
