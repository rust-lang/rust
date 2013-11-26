// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:xcrate_static_addresses.rs

extern mod xcrate_static_addresses;

use other = xcrate_static_addresses;

pub fn main() {
    other::verify_same(&other::global);
    other::verify_same2(other::global2);

    // Previously this fail'd because there were two addresses that were being
    // used when declaring constants.
    other::test::cond.trap(|_| {
    }).inside(|| {
        other::raise();
    })
}
