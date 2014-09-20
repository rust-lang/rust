// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we can use `-Z lto` when linking against libraries that were
// separately compiled.

// aux-build:sepcomp_lib.rs
// compile-flags: -Z lto
// no-prefer-dynamic

extern crate sepcomp_lib;
use sepcomp_lib::a::one;
use sepcomp_lib::b::two;
use sepcomp_lib::c::three;

fn main() {
    assert_eq!(one(), 1);
    assert_eq!(two(), 2);
    assert_eq!(three(), 3);
}
