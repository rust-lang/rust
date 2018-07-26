// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    // Test that these type check correctly.
    (&42u8 >> 4) as usize;
    (&42u8 << 4) as usize;

    let cap = 512 * 512;
    cap as u8;
    // Assert `cap` did not get inferred to `u8` and overflowed.
    assert_ne!(cap, 0);
}
