// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub fn foobar() -> int { 1 }
}

mod b {
    pub fn foobar() -> int { 2 }
}

mod c {
    // Technically the second use shadows the first, but in theory it should
    // only be shadowed for this module. The implementation of resolve currently
    // doesn't implement this, so this test is ensuring that using "c::foobar"
    // is *not* getting b::foobar. Today it's an error, but perhaps one day it
    // can correctly get a::foobar instead.
    pub use a::foobar;
    use b::foobar;
}

fn main() {
    assert_eq!(c::foobar(), 1);
    //~^ ERROR: unresolved name `c::foobar`
}

