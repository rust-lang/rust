// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #14893. Tests that casts from vectors don't behave strangely in the
// presence of the `_` type shorthand notation.

struct X {
    y: [u8, ..2],
}

fn main() {
    let x1 = X { y: [0, 0] };

    let p1: *u8 = &x1.y as *_;  //~ ERROR mismatched types
    let t1: *[u8, ..2] = &x1.y as *_;
    let h1: *[u8, ..2] = &x1.y as *[u8, ..2];

    let mut x1 = X { y: [0, 0] };

    let p1: *mut u8 = &mut x1.y as *mut _;  //~ ERROR mismatched types
    let t1: *mut [u8, ..2] = &mut x1.y as *mut _;
    let h1: *mut [u8, ..2] = &mut x1.y as *mut [u8, ..2];
}

