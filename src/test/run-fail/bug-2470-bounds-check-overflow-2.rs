// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test
// error-pattern:index out of bounds

use std::uint;

fn main() {
    let x = vec!(1_usize,2_usize,3_usize);

    // This should cause a bounds-check panic, but may not if we do our
    // bounds checking by comparing a scaled index value to the vector's
    // length (in bytes), because the scaling of the index will cause it to
    // wrap around to a small number.

    let idx = uint::MAX & !(uint::MAX >> 1_usize);
    println!("ov2 idx = 0x%x", idx);

    // This should panic.
    println!("ov2 0x%x",  x[idx]);
}
