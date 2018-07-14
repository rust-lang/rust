// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test behavior of `?` macro _separator_ under the 2015 edition. Namely, `?` can be used as a
// separator, but you get a migration warning for the edition.

// compile-flags: --edition=2015
// compile-pass

#![warn(rust_2018_compatibility)]

macro_rules! bar {
    ($(a)?*) => {} //~WARN using `?` as a separator
    //~^WARN this was previously accepted
}

macro_rules! baz {
    ($(a)?+) => {} //~WARN using `?` as a separator
    //~^WARN this was previously accepted
}

fn main() {
    bar!();
    bar!(a);
    bar!(a?a);
    bar!(a?a?a?a?a);

    baz!(a);
    baz!(a?a);
    baz!(a?a?a?a?a);
}
