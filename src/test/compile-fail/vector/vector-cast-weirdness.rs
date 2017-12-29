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
// Update: after a change to the way casts are done, we have more type information
// around and so the errors here are no longer exactly the same.

struct X {
    y: [u8; 2],
}

fn main() {
    let x1 = X { y: [0, 0] };

    // No longer a type mismatch - the `_` can be fully resolved by type inference.
    let p1: *const u8 = &x1.y as *const _;
    let t1: *const [u8; 2] = &x1.y as *const _;
    let h1: *const [u8; 2] = &x1.y as *const [u8; 2];

    let mut x1 = X { y: [0, 0] };

    // This is still an error since we don't allow casts from &mut [T; n] to *mut T.
    let p1: *mut u8 = &mut x1.y as *mut _;  //~ ERROR casting
    let t1: *mut [u8; 2] = &mut x1.y as *mut _;
    let h1: *mut [u8; 2] = &mut x1.y as *mut [u8; 2];
}
