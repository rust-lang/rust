// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C debug-assertions

#![feature(core_simd)]

use std::simd::u32x4;

// (Work around constant-evaluation)
fn id<T>(x: T) -> T { x }

fn eq_u32x4(u32x4(x0, x1, x2, x3): u32x4, u32x4(y0, y1, y2, y3): u32x4) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
}

fn main() {
    assert!(eq_u32x4(u32x4(1, 1, 1, 1) << id(u32x4(1, 1, 1, 1)), u32x4(2, 2, 2, 2)));
}
